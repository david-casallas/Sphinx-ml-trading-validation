"""
test_mt5_regime.py

Smoke test for mt5_wrapper.py and regime_filter.py.

Verifies:
1. MT5 connection works with your account credentials
2. USA500 historical bars are accessible
3. VXX.US daily bars are accessible
4. Regime filter calculates correctly
5. Current tick can be read

Does NOT place any orders. Pure read-only test.

USAGE:
    cd src
    python test_mt5_regime.py
    
    Will prompt for your MT5 credentials interactively (so they're not in a file).
"""

from __future__ import annotations
import sys
import getpass

import pytest

# MetaTrader5 is a Windows-only package. On Linux/macOS this whole module is
# skipped so `pytest` runs green from a fresh clone; the offline tests
# (test_risk_manager.py, test_feature_parity.py) cover everything that does
# not require a live MT5 terminal.
pytest.importorskip(
    "MetaTrader5",
    reason="MetaTrader5 is Windows-only; live-connection smoke test skipped",
)

from mt5_wrapper import MT5Wrapper, MT5Error
from regime_filter import RegimeFilter


def main():
    print('=' * 65)
    print('  MT5 + REGIME FILTER SMOKE TEST')
    print('=' * 65)
    
    # === 1. Credentials ===
    print('\n[1/5] MT5 credentials')
    login = int(input('  Login (account number): ').strip())
    server = input('  Server (e.g. ActivTradesMarkets-Server): ').strip()
    password = getpass.getpass('  Password (hidden): ')
    
    # === 2. Connect ===
    print('\n[2/5] Connecting to MT5...')
    wrapper = MT5Wrapper(login=login, password=password, server=server)
    try:
        wrapper.connect()
    except MT5Error as e:
        print(f'  ✗ Connection failed: {e}')
        return 1
    
    # === 3. Test Usa500 bars ===
    print('\n[3/5] Fetching Usa500 M15 bars...')
    try:
        df_usa = wrapper.get_bars('Usa500', 'M15', 300)
        print(f'  ✓ Got {len(df_usa)} bars')
        print(f'    First: {df_usa.iloc[0]["Datetime"]}')
        print(f'    Last:  {df_usa.iloc[-1]["Datetime"]}')
        print(f'    Last close: {df_usa.iloc[-1]["Close"]:.2f}')
        
        tick = wrapper.get_current_tick('Usa500')
        print(f'  ✓ Current tick: bid={tick["bid"]:.2f}, ask={tick["ask"]:.2f}, spread={tick["spread"]:.2f}')
    except MT5Error as e:
        print(f'  ✗ USA500 fetch failed: {e}')
        wrapper.disconnect()
        return 1
    
    # === 4. Test VXX.US bars ===
    print('\n[4/5] Fetching VXX.US daily bars...')
    try:
        df_vxx = wrapper.get_bars('VXX.US', 'D1', 250)
        print(f'  ✓ Got {len(df_vxx)} daily bars')
        print(f'    First: {df_vxx.iloc[0]["Datetime"].date()}')
        print(f'    Last:  {df_vxx.iloc[-1]["Datetime"].date()}')
        print(f'    Last close: {df_vxx.iloc[-1]["Close"]:.2f}')
    except MT5Error as e:
        print(f'  ✗ VXX.US fetch failed: {e}')
        print(f'    Make sure VXX.US is in your Market Watch (right-click → Show All)')
        wrapper.disconnect()
        return 1
    
    # === 5. Regime filter ===
    print('\n[5/5] Computing regime filter...')
    try:
        regime = RegimeFilter(
            mt5_wrapper=wrapper,
            symbol='VXX.US',
            sma_short=20,
            sma_long=200,
            threshold_diff=10.0,
        )
        can_trade = regime.can_trade()
        details = regime.get_details()
        
        print(f'  ✓ Regime computed')
        print(f'    {regime.status_line()}')
        print(f'\n  Detail:')
        print(f'    can_trade:        {details["can_trade"]}')
        print(f'    high_uncertainty: {details["high_uncertainty"]}')
        print(f'    VXX close (yest): {details["vxx_close"]:.2f}')
        print(f'    SMA20:            {details["sma_short"]:.2f}')
        print(f'    SMA200:           {details["sma_long"]:.2f}')
        print(f'    diff:             {details["diff"]:+.2f}')
        print(f'    threshold:        {details["threshold"]:.2f}')
    except (MT5Error, Exception) as e:
        print(f'  ✗ Regime filter failed: {e}')
        wrapper.disconnect()
        return 1
    
    # === Done ===
    wrapper.disconnect()
    print('\n' + '=' * 65)
    print('  ✓ ALL TESTS PASSED')
    print('=' * 65)
    print('\n  Both mt5_wrapper.py and regime_filter.py are working correctly.')
    print('  Ready to integrate into the main bot.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
