"""
mt5_wrapper.py

Thin wrapper around MetaTrader5 Python API.

Centralizes:
- Connection lifecycle (init, login, shutdown)
- Symbol selection and validation
- Historical data fetching (bars, ticks)
- Order placement and position management
- Error handling with clear messages

Why a wrapper:
- Isolates MT5 API quirks from the rest of the bot
- Easier to mock for unit tests
- Single place to handle reconnection logic
"""

from __future__ import annotations
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import time
import pandas as pd
import MetaTrader5 as mt5


class MT5Error(Exception):
    """Raised when MT5 operations fail."""
    pass


class MT5Wrapper:
    """
    Manages connection and operations against a single MT5 account.
    
    Usage:
        wrapper = MT5Wrapper(login=12345, password="...", server="...")
        wrapper.connect()
        df = wrapper.get_bars("USA500", "M15", 1000)
        wrapper.disconnect()
    """
    
    # Timeframe mapping
    TIMEFRAMES = {
        'M1':  mt5.TIMEFRAME_M1,
        'M5':  mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1':  mt5.TIMEFRAME_H1,
        'H4':  mt5.TIMEFRAME_H4,
        'D1':  mt5.TIMEFRAME_D1,
    }
    
    def __init__(self, login: int, password: str, server: str, path: Optional[str] = None):
        self.login = int(login)
        self.password = password
        self.server = server
        self.path = path
        self._connected = False
    
    def connect(self) -> None:
        """Initialize MT5 and login. Raises MT5Error on failure."""
        kwargs = {}
        if self.path:
            kwargs['path'] = self.path
        
        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            raise MT5Error(f'mt5.initialize() failed: {err}')
        
        if not mt5.login(self.login, password=self.password, server=self.server):
            err = mt5.last_error()
            mt5.shutdown()
            raise MT5Error(f'mt5.login() failed for {self.login}@{self.server}: {err}')
        
        self._connected = True
        info = mt5.account_info()
        if info is None:
            raise MT5Error('Connected but cannot read account_info')
        
        print(f'[MT5] Connected: account={info.login}, balance={info.balance:.2f} {info.currency}, server={info.server}')
    
    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False
    
    def ensure_symbol(self, symbol: str) -> None:
        """Make sure symbol is selected in Market Watch and visible."""
        info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f'Symbol {symbol} not found in MT5')
        
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise MT5Error(f'Failed to select symbol {symbol}')
            time.sleep(0.5)  # let MT5 register the selection
    
    def get_bars(self, symbol: str, timeframe: str, count: int,
                 end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch the last N closed bars for a symbol.
        
        IMPORTANT: returns CLOSED bars only. The currently forming bar
        is NOT included. This matches the backtest convention of using
        information up to the previous bar's close.
        
        Returns DataFrame with columns: Datetime, Open, High, Low, Close, TickVol, Spread
        """
        self.ensure_symbol(symbol)
        
        tf = self.TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            raise MT5Error(f'Unsupported timeframe: {timeframe}')
        
        if end_time is None:
            end_time = datetime.now()
        
        rates = mt5.copy_rates_from(symbol, tf, end_time, count)
        if rates is None or len(rates) == 0:
            raise MT5Error(f'No bars returned for {symbol} {timeframe}')
        
        df = pd.DataFrame(rates)
        # Use utc=True to ignore machine timezone, then strip tz to get naive broker time.
        # MT5 encodes server local time (Madrid) into the Unix timestamp as if it were UTC.
        # This makes the result correct on any machine regardless of system TZ.
        df['Datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'tick_volume': 'TickVol', 'spread': 'Spread'
        })
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Spread']]
        df = df.sort_values('Datetime').reset_index(drop=True)
        return df
    
    def get_current_tick(self, symbol: str) -> Dict:
        """Get the current bid/ask tick."""
        self.ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(f'No tick for {symbol}')
        return {
            'bid': tick.bid, 'ask': tick.ask, 'last': tick.last,
            'time': datetime.utcfromtimestamp(tick.time),  # broker time
            'spread': (tick.ask - tick.bid),
        }
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol metadata (point, volume limits, etc.)."""
        self.ensure_symbol(symbol)
        info = mt5.symbol_info(symbol)
        return {
            'name': info.name,
            'point': info.point,
            'digits': info.digits,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_contract_size': info.trade_contract_size,
        }
    
    def place_order(self, symbol: str, order_type: str, volume: float,
                    sl_price: float, tp_price: float,
                    magic: int, comment: str = '',
                    deviation: int = 20) -> Dict:
        """
        Place a market order with TP and SL.
        
        order_type: 'BUY' or 'SELL'
        """
        self.ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        
        if order_type.upper() == 'BUY':
            mt5_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif order_type.upper() == 'SELL':
            mt5_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            raise MT5Error(f'Invalid order_type: {order_type}')
        
        # Round prices to symbol's digits
        digits = info.digits
        price = round(price, digits)
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': mt5_type,
            'price': price,
            'sl': sl_price,
            'tp': tp_price,
            'deviation': deviation,
            'magic': magic,
            'comment': comment[:31],  # MT5 limit
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result is None:
            raise MT5Error(f'order_send returned None: {mt5.last_error()}')
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Try IOC if FOK failed
            request['type_filling'] = mt5.ORDER_FILLING_IOC
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise MT5Error(f'Order failed: retcode={result.retcode}, comment={result.comment}')
        
        return {
            'ticket': result.order,
            'price': result.price,
            'volume': result.volume,
            'sl': sl_price,
            'tp': tp_price,
            'comment': comment,
            'time': datetime.now(),
        }
    
    def get_open_positions(self, magic: Optional[int] = None) -> List[Dict]:
        """List currently open positions, optionally filtered by magic number."""
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        result = []
        for p in positions:
            if magic is not None and p.magic != magic:
                continue
            result.append({
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': p.volume,
                'price_open': p.price_open,
                'price_current': p.price_current,
                'sl': p.sl,
                'tp': p.tp,
                'profit': p.profit,
                'magic': p.magic,
                'comment': p.comment,
                'time_open': datetime.utcfromtimestamp(p.time),  # broker time
            })
        return result
    
    def close_position(self, ticket: int, deviation: int = 20) -> Dict:
        """Close a specific position by ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise MT5Error(f'Position {ticket} not found')
        p = positions[0]
        
        tick = mt5.symbol_info_tick(p.symbol)
        if p.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': p.symbol,
            'volume': p.volume,
            'type': close_type,
            'position': ticket,
            'price': price,
            'deviation': deviation,
            'magic': p.magic,
            'comment': 'TIMEOUT_CLOSE',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            request['type_filling'] = mt5.ORDER_FILLING_IOC
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise MT5Error(f'Close failed for ticket {ticket}: {result.retcode if result else "None"}')
        
        return {'ticket': ticket, 'close_price': result.price, 'time': datetime.now()}
