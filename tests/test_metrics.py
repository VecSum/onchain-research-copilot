"""Metric tests.

The golden values are the point of this file: each metric is checked against a
known series, so a refactor that quietly changes a formula fails here instead of
shipping a number that looks plausible and is wrong. Tolerances are tight where a
closed form exists and loose only where the definition is a convention.
"""

from __future__ import annotations

import math
import statistics
import unittest

from onchain_copilot.metrics import (
    closes,
    compute,
    max_drawdown,
    realized_volatility,
    return_zscore,
    risk_flags,
    rsi,
    simple_returns,
    upside_downside_capture,
    volume_ratio,
)
from onchain_copilot.models import Candle, Snapshot


def series(prices, volume=1000.0, start=1756680000, step=3600):
    """Build candles from close prices with a fixed OHLC shape."""

    candles = []
    for index, price in enumerate(prices):
        candles.append(Candle(ts=start + index * step, open=price * 0.999,
                               high=price * 1.01, low=price * 0.99,
                               close=price, volume=volume))
    return candles


FLAT = series([1.0] * 30)

RISING = [1.0 + 0.01 * index for index in range(30)]

MIXED = [1.0, 1.05, 0.98, 1.10, 0.95, 1.04, 1.12, 0.92, 1.06, 1.00,
         1.08, 0.97, 1.14, 1.02, 0.99, 1.11, 1.03, 0.94, 1.07, 1.05,
         1.09, 0.93, 1.15, 1.01, 1.13, 1.00, 1.06, 0.96, 1.16, 1.04]


class TestReturns(unittest.TestCase):
    def test_simple_returns_are_relative(self):
        returns = simple_returns(series([1.0, 1.1, 0.99]))
        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns[0], 0.1)
        self.assertAlmostEqual(returns[1], -0.1)

    def test_flat_series_has_no_volatility(self):
        self.assertIsNone(realized_volatility(FLAT, 24))

    def test_monotone_series_has_volatility(self):
        # Alternating +/-10% returns: the sample variance has a closed form, so
        # the annualised expectation is derived instead of remembered.
        returns = [0.10 if index % 2 == 0 else -0.10 for index in range(24)]
        price = 1.0
        candles = []
        for index, ret in enumerate(returns):
            price *= 1.0 + ret
            candles.append(Candle(ts=1756680000 + index * 3600, open=price,
                                  high=price, low=price, close=price,
                                  volume=1000.0))
        value = realized_volatility(candles, 24)
        expected = statistics.stdev(returns) * math.sqrt(24 * 365) * 100.0
        self.assertIsNotNone(value)
        self.assertGreater(value, 800.0)
        self.assertAlmostEqual(value, expected, places=2)

    def test_insufficient_data_returns_none(self):
        self.assertIsNone(realized_volatility(series([1.0, 1.01, 1.02]), 24))


class TestRSI(unittest.TestCase):
    def test_monotone_rise_is_100(self):
        self.assertEqual(rsi(series(RISING), 14), 100.0)

    def test_flat_series_is_fifty(self):
        self.assertEqual(rsi(FLAT, 14), 50.0)

    def test_short_series_is_none(self):
        self.assertIsNone(rsi(series([1.0, 1.01, 1.02]), 14))

    def test_rsi_stays_in_range(self):
        value = rsi(series(MIXED), 14)
        self.assertTrue(0.0 <= value <= 100.0)


class TestDrawdown(unittest.TestCase):
    def test_known_peak_to_trough(self):
        value = max_drawdown(series([100.0, 120.0, 90.0, 130.0]))
        self.assertAlmostEqual(value, 25.0, places=4)

    def test_monotone_rise_has_zero_drawdown(self):
        self.assertEqual(max_drawdown(series(RISING)), 0.0)

    def test_empty_is_none(self):
        self.assertIsNone(max_drawdown([]))


class TestVolumeZAndCapture(unittest.TestCase):
    def test_volume_ratio_compares_to_trailing_mean(self):
        candles = series([1.0] * 25, volume=100.0)
        candles[-1] = Candle(candles[-1].ts, 1.0, 1.0, 1.0, 1.0, 400.0)
        ratio = volume_ratio(candles)
        self.assertAlmostEqual(ratio, 4.0, places=4)

    def test_zscore_of_typical_move_is_small(self):
        value = return_zscore(series(MIXED), 24)
        self.assertIsNotNone(value)
        self.assertTrue(abs(value) < 3.0)

    def test_capture_is_none_without_moves(self):
        self.assertEqual(upside_downside_capture(FLAT), (None, None))


class TestCompute(unittest.TestCase):
    def test_never_raises_on_empty_input(self):
        result = compute([], None)
        self.assertIn("insufficient_candles", result.flags)
        self.assertTrue(all(metric.value is None for metric in result.metrics))

    def test_snapshot_metrics_use_snapshot_fields(self):
        snapshot = Snapshot(pool_key="bsc:0xabc", symbol="ABC", chain="bsc",
                              ts="2026-09-01T00:00:00Z", price_usd=2.0,
                              liquidity_usd=100_000.0, volume_24h_usd=300_000.0,
                              fdv_usd=1_000_000.0, source="fixtures")
        result = compute(series(RISING), snapshot)
        values = {metric.name: metric.value for metric in result.metrics}
        self.assertAlmostEqual(values["volume_per_liquidity"], 3.0, places=4)
        self.assertAlmostEqual(values["fdv_to_liquidity"], 10.0, places=4)

    def test_thin_liquidity_flagged(self):
        snapshot = Snapshot(pool_key="bsc:0xabc", symbol="ABC", chain="bsc",
                              ts="2026-09-01T00:00:00Z", price_usd=2.0,
                              liquidity_usd=1_000.0, volume_24h_usd=9_000.0,
                              fdv_usd=500_000.0, source="fixtures")
        result = compute(series(RISING), snapshot)
        values = {metric.name: metric.value for metric in result.metrics}
        self.assertIn("thin_liquidity", risk_flags(values))

    def test_deep_drawdown_flagged(self):
        result = compute(series([1.0, 2.0, 0.7, 0.9]), None)
        values = {metric.name: metric.value for metric in result.metrics}
        self.assertIn("deep_drawdown", risk_flags(values))

    def test_metrics_are_pure(self):
        first = compute(series(MIXED), None)
        second = compute(series(MIXED), None)
        self.assertEqual([m.value for m in first.metrics],
                         [m.value for m in second.metrics])


if __name__ == "__main__":
    unittest.main()
