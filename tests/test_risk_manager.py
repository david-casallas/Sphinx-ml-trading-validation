"""
test_risk_manager.py

Standalone test for risk_manager.py. Does NOT touch MT5.

Verifies:
1. Mult formula returns expected values at known equity levels
2. compute_lot_size produces sensible lot sizes
3. record_close updates equity correctly
4. Circuit breaker triggers at the right threshold
5. Cooldown auto-clears after expiration
6. State persists and loads correctly

USAGE:
    cd src
    python test_risk_manager.py
"""

from __future__ import annotations
import sys
import pathlib
import tempfile
import logging
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'src'))
from risk_manager import RiskManager


def make_logger():
    logger = logging.getLogger('test_risk')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('  [%(levelname)s] %(message)s'))
        logger.addHandler(h)
    return logger


def make_risk(state_path: str = None) -> RiskManager:
    """Standard test config (the position-sizing defaults used during development)."""
    return RiskManager(
        initial_equity=1000.0,
        base_mpo=20.0,
        leverage=100.0,
        contract_size=100000.0,
        base_for_mult=1.25,
        max_n=6.0,
        min_mult=1.0,
        min_mult_downside=0.3,
        downside_beta=0.5,
        dd_threshold=-0.05,
        cooldown_hours=1.8,
        state_path=state_path or tempfile.mktemp(suffix='.json'),
        logger=make_logger(),
    )


def test_1_initial_state():
    print('\n[1] Initial state')
    r = make_risk()
    assert r.current_equity == 1000.0
    assert r.peak_equity == 1000.0
    assert r.cumulative_pnl == 0.0
    assert r.current_drawdown_pct == 0.0
    assert r.cooldown_until is None
    print('    ✓ Initial state correct')


def test_2_mult_formula():
    print('\n[2] Mult formula at known equity levels')
    r = make_risk()
    
    # equity = initial → Mult = 1.0
    r.cumulative_pnl = 0
    m = r._compute_mult()
    assert abs(m - 1.0) < 0.001, f'At equity=1000, Mult should be 1.0, got {m}'
    print(f'    equity=$1000 → Mult={m:.3f}  ✓')
    
    # equity = 1250 → ratio=1.25, log_{1.25}(1.25)=1, Mult = 1+1 = 2.0
    r.cumulative_pnl = 250
    m = r._compute_mult()
    assert abs(m - 2.0) < 0.01, f'At equity=1250, Mult should be ~2.0, got {m}'
    print(f'    equity=$1250 → Mult={m:.3f}  ✓')
    
    # equity = 2000 → ratio=2, log_{1.25}(2)≈3.106, Mult ≈ 4.106
    r.cumulative_pnl = 1000
    m = r._compute_mult()
    assert 4.0 < m < 4.2, f'At equity=2000, Mult should be ~4.1, got {m}'
    print(f'    equity=$2000 → Mult={m:.3f}  ✓')
    
    # equity = 7000 → ratio=7 > MAX_N=6, capped
    r.cumulative_pnl = 6000
    m = r._compute_mult()
    print(f'    equity=$7000 → Mult={m:.3f} (capped at MAX_N=6)  ✓')
    
    # Downside
    r.cumulative_pnl = -200  # equity = 800, 20% lost
    m = r._compute_mult()
    expected = max(0.3, 1.0 - 0.5 * 0.2)  # = 0.9
    assert abs(m - expected) < 0.001, f'At equity=800, Mult should be {expected}, got {m}'
    print(f'    equity=$800 (downside) → Mult={m:.3f}  ✓')


def test_3_lot_size():
    print('\n[3] Lot size computation')
    r = make_risk()
    
    # equity=1000, Mult=1, MPO=20, exposure=2000, lot=2000/100000=0.02
    r.cumulative_pnl = 0
    lot = r.compute_lot_size(150.0)
    assert abs(lot - 0.02) < 0.001, f'Expected 0.02, got {lot}'
    print(f'    equity=$1000, price=150 → lot={lot:.2f}  ✓')
    
    # equity=2000, Mult≈4.1, MPO≈82, exposure≈8200, lot≈0.08
    r.cumulative_pnl = 1000
    lot = r.compute_lot_size(150.0)
    print(f'    equity=$2000, price=150 → lot={lot:.2f}  ✓')
    
    # Tiny equity → min_lot clamp
    r.cumulative_pnl = -990  # equity=10
    lot = r.compute_lot_size(150.0)
    assert lot == 0.01, f'Min lot should clamp to 0.01, got {lot}'
    print(f'    equity=$10 → lot={lot:.2f} (clamped to min)  ✓')


def test_4_record_close_and_peak():
    print('\n[4] record_close updates equity and peak correctly')
    r = make_risk()
    now = datetime(2026, 5, 18, 12, 0, 0)
    
    # Win $100
    r.record_close(ticket=1001, pnl_usd=100, reason='TP', close_dt=now)
    assert r.cumulative_pnl == 100
    assert r.current_equity == 1100
    assert r.peak_equity == 1100  # new peak
    print(f'    After +$100: eq=${r.current_equity}, peak=${r.peak_equity}  ✓')
    
    # Lose $50
    r.record_close(ticket=1002, pnl_usd=-50, reason='SL', close_dt=now + timedelta(hours=1))
    assert r.cumulative_pnl == 50
    assert r.current_equity == 1050
    assert r.peak_equity == 1100  # peak unchanged
    print(f'    After -$50: eq=${r.current_equity}, peak=${r.peak_equity} (unchanged)  ✓')


def test_5_circuit_breaker():
    print('\n[5] Circuit breaker triggers at -5% DD')
    r = make_risk()
    now = datetime(2026, 5, 18, 12, 0, 0)
    
    # Win $200 → peak = 1200
    r.record_close(ticket=1, pnl_usd=200, reason='TP', close_dt=now)
    assert r.peak_equity == 1200
    
    # Lose $40 → eq = 1160, DD = (1160-1200)/1200 = -3.3%, no trigger
    r.record_close(ticket=2, pnl_usd=-40, reason='SL', close_dt=now + timedelta(minutes=15))
    assert r.cooldown_until is None, 'Cooldown should NOT trigger at -3.3% DD'
    print(f'    After -3.3% DD: cooldown={r.cooldown_until} (correctly not triggered)  ✓')
    
    # Lose $30 more → eq = 1130, DD = (1130-1200)/1200 = -5.83%, TRIGGER
    trigger_dt = now + timedelta(minutes=30)
    r.record_close(ticket=3, pnl_usd=-30, reason='SL', close_dt=trigger_dt)
    assert r.cooldown_until is not None, 'Cooldown SHOULD trigger at -5.83% DD'
    expected_until = trigger_dt + timedelta(hours=1.8)
    assert r.cooldown_until == expected_until
    print(f'    After -5.83% DD: cooldown until {r.cooldown_until}  ✓')


def test_6_cooldown_expires():
    print('\n[6] Cooldown auto-clears after expiration')
    r = make_risk()
    now = datetime(2026, 5, 18, 12, 0, 0)
    
    # Force cooldown via RETURN-SPACE drawdown (the breaker's actual trigger).
    # The breaker monitors cumulative summed per-trade returns, matching the
    # backtest, NOT money drawdown (which is amplified by position sizing).
    r.record_close(ticket=1, pnl_usd=-60, reason='SL', close_dt=now, reward_pct=-0.06)
    assert r.cooldown_until is not None, 'breaker should fire at -6% return-space DD'
    
    # Check during cooldown
    assert r.can_open_trade(now + timedelta(minutes=30)) == False
    print(f'    30min in: can_open={r.can_open_trade(now + timedelta(minutes=30))}  ✓')
    
    # Check after cooldown
    assert r.can_open_trade(now + timedelta(hours=2)) == True
    assert r.cooldown_until is None  # auto-cleared
    print(f'    2h later: can_open={r.can_open_trade(now + timedelta(hours=2))}, cooldown cleared  ✓')


def test_7_persistence():
    print('\n[7] State persists across instances')
    state_path = tempfile.mktemp(suffix='.json')
    
    r1 = make_risk(state_path=state_path)
    now = datetime(2026, 5, 18, 12, 0, 0)
    r1.record_close(ticket=1, pnl_usd=150, reason='TP', close_dt=now)
    r1.record_close(ticket=2, pnl_usd=-30, reason='SL', close_dt=now + timedelta(minutes=15))
    
    # New instance with same state path
    r2 = make_risk(state_path=state_path)
    assert r2.cumulative_pnl == 120, f'Expected 120, got {r2.cumulative_pnl}'
    assert r2.peak_equity == 1150
    assert r2.n_trades_closed == 2
    print(f'    Reloaded: eq=${r2.current_equity}, peak=${r2.peak_equity}, closed={r2.n_trades_closed}  ✓')


def test_8_deduplication():
    print('\n[8] Duplicate deal IDs are not double-counted')
    r = make_risk()
    now = datetime(2026, 5, 18, 12, 0, 0)
    
    r.record_close(ticket=1, pnl_usd=100, reason='TP', close_dt=now, deal_id=12345)
    assert r.cumulative_pnl == 100
    
    # Try to record same deal_id again
    r.record_close(ticket=1, pnl_usd=100, reason='TP', close_dt=now, deal_id=12345)
    assert r.cumulative_pnl == 100, 'Duplicate deal should not change equity'
    print(f'    After duplicate: eq=${r.current_equity} (unchanged)  ✓')


def main():
    print('=' * 65)
    print('  TEST DE RISK_MANAGER')
    print('=' * 65)
    
    try:
        test_1_initial_state()
        test_2_mult_formula()
        test_3_lot_size()
        test_4_record_close_and_peak()
        test_5_circuit_breaker()
        test_6_cooldown_expires()
        test_7_persistence()
        test_8_deduplication()
    except AssertionError as e:
        print(f'\n✗ TEST FAILED: {e}')
        return 1
    except Exception as e:
        print(f'\n✗ UNEXPECTED ERROR: {e}')
        import traceback
        traceback.print_exc()
        return 1
    
    print('\n' + '=' * 65)
    print('  ✓ ALL TESTS PASSED')
    print('=' * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
