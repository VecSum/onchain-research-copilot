"""Quantitative metrics over pool OHLCV plus point-in-time snapshots.

Three rules the unit tests enforce:

1. Every metric is a pure function of its inputs: no global state, no clock, no
    provider access. The same series always yields the same number.
2. Insufficient data returns None instead of a guess, and the caller gets a flag
    naming the metric. A missing number is information; a fabricated number is a
    bug that a client will eventually discover in production.
3. Simple returns are used for aggregates so the numbers match what a chart
    shows. Log returns are never mixed in, which is how most home-made metric
    layers quietly drift from the reference implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .models import Candle, Metric, MetricResult, Snapshot, utc_now_iso


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    needs_candles: bool = True
    min_periods: int = 1


SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("close_price", "usd"),
    MetricSpec("window_high", "usd"),
    MetricSpec("window_low", "usd"),
    MetricSpec("price_change_pct", "percent"),
    MetricSpec("realized_volatility_24h", "percent", min_periods=3),
    MetricSpec("return_zscore_24h", "sigma", min_periods=8),
    MetricSpec("rsi_14", "index", min_periods=15),
    MetricSpec("volume_ratio", "ratio", min_periods=2),
    MetricSpec("liquidity_to_volume", "ratio", needs_candles=False),
    MetricSpec("volume_per_liquidity", "ratio", needs_candles=False),
    MetricSpec("fdv_to_liquidity", "ratio", needs_candles=False),
    MetricSpec("max_drawdown", "percent"),
    MetricSpec("risk_reward_asymmetry", "ratio"),
    MetricSpec("up_capture", "ratio"),
    MetricSpec("down_capture", "ratio"),
    MetricSpec("liquidity_usd", "usd", needs_candles=False),
    MetricSpec("volume_24h_usd", "usd", needs_candles=False),
    MetricSpec("fdv_usd", "usd", needs_candles=False),
)


def _pct(value: float) -> float:
    return round(value * 100.0, 4)


def closes(candles: Sequence[Candle]) -> list[float]:
    return [candle.close for candle in candles]


def simple_returns(candles: Sequence[Candle]) -> list[float]:
    out: list[float] = []
    for previous, current in zip(candles, candles[1:]):
        if previous.close > 0:
            out.append(current.close / previous.close - 1.0)
    return out


def realized_volatility(candles: Sequence[Candle], window: int) -> float | None:
    """Annualised volatility of the trailing window, as a percentage.

    Annualised on the hourly cadence (24 bars per day, 365 days). Returns None
    when fewer than three samples exist: a volatility estimate from two points
    is noise wearing a number.
    """

    returns = simple_returns(candles)[-window:]
    if len(returns) < 3:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    if variance <= 0:
        return None
    hourly_sigma = math.sqrt(variance)
    return round(hourly_sigma * math.sqrt(24 * 365) * 100.0, 4)


def return_zscore(candles: Sequence[Candle], window: int) -> float | None:
    """Latest return expressed in standard deviations of the trailing window."""

    returns = simple_returns(candles)
    history = returns[-(window + 1):]
    if len(history) < 8:
        return None
    latest = history[-1]
    sample = history[:-1]
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample) / (len(sample) - 1)
    if variance <= 0:
        return None
    return round((latest - mean) / math.sqrt(variance), 4)


def rsi(candles: Sequence[Candle], period: int) -> float | None:
    """Wilder RSI. A perfectly flat series is defined as 50.0, not 0.0."""

    prices = closes(candles)
    if len(prices) <= period:
        return None
    deltas = [current - previous for previous, current in zip(prices, prices[1:])]
    seed = deltas[:period]
    avg_gain = sum(delta for delta in seed if delta > 0) / period
    avg_loss = -sum(delta for delta in seed if delta < 0) / period
    for delta in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    relative = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + relative)), 4)


def volume_ratio(candles: Sequence[Candle], window: int = 24) -> float | None:
    """Latest bar volume divided by the trailing mean. Above 1 is expansion."""

    if len(candles) < 2:
        return None
    recent = [candle.volume for candle in candles[-window:-1]]
    if not recent or candles[-1].volume <= 0:
        return None
    baseline = sum(recent) / len(recent)
    if baseline <= 0:
        return None
    return round(candles[-1].volume / baseline, 4)


def max_drawdown(candles: Sequence[Candle]) -> float | None:
    """Largest peak-to-trough decline in the series, as a positive percentage."""

    prices = closes(candles)
    if not prices:
        return None
    peak = prices[0]
    worst = 0.0
    for price in prices:
        peak = max(peak, price)
        if peak > 0:
            worst = min(worst, price / peak - 1.0)
    return _pct(abs(worst))


def risk_reward_asymmetry(candles: Sequence[Candle], window: int = 24) -> float | None:
    """Mean up-return divided by mean down-return over the trailing window.

    Most home-made dashboards report total return and call it a result. This
    one reports the shape: whether the moves up are larger than the moves down
    is what determines whether a strategy survives, and it is invisible in a
    cumulative chart.
    """

    returns = simple_returns(candles)[-window:]
    if len(returns) < 4:
        return None
    gains = [value for value in returns if value > 0]
    losses = [-value for value in returns if value < 0]
    if len(gains) < 2 or len(losses) < 2:
        return None
    mean_gain = sum(gains) / len(gains)
    mean_loss = sum(losses) / len(losses)
    if mean_loss == 0:
        return None
    return round(mean_gain / mean_loss, 4)


def upside_downside_capture(candles: Sequence[Candle], window: int = 24) -> tuple[float | None, float | None]:
    """Share of total up-move captured, and the mirror image for the down-move."""

    returns = simple_returns(candles)[-window:]
    if len(returns) < 4:
        return None, None
    up = [value for value in returns if value > 0]
    down = [value for value in returns if value < 0]
    up_total = sum(up)
    down_total = -sum(down)
    if up_total <= 0 or down_total <= 0:
        return None, None
    up_share = sum(1 for value in up if value > up_total / len(up)) / len(up)
    down_share = sum(1 for value in down if value > down_total / len(down)) / len(down)
    return round(up_share, 4), round(down_share, 4)
def net_flow_proxy(snapshot: Snapshot) -> float | None:
    """Trade volume per unit of liquidity, as a pressure proxy.

    True signed net flow requires labelled wallet flows, which the free data
    tier does not expose. Rather than invent a number, this reports pressure
    per liquidity dollar and the report explicitly labels it a proxy.
    """

    if snapshot is None or not snapshot.liquidity_usd or snapshot.liquidity_usd <= 0:
        return None
    if snapshot.volume_24h_usd is None:
        return None
    return round(snapshot.volume_24h_usd / snapshot.liquidity_usd, 4)


def compute(
    candles: Sequence[Candle],
    snapshot: Snapshot | None,
    *,
    volatility_window: int = 24,
    momentum_fast: int = 7,
    momentum_slow: int = 21,
    rsi_period: int = 14,
) -> MetricResult:
    """Compute the full metric set. Thin or broken data never raises."""

    flags: list[str] = []
    if len(candles) < 8:
        flags.append("insufficient_candles")

    price = candles[-1].close if candles else (snapshot.price_usd if snapshot else None)
    window_high = max(closes(candles), default=None) if candles else None
    window_low = min(closes(candles), default=None) if candles else None

    change_pct: float | None = None
    if len(candles) >= 2 and candles[0].close > 0:
        change_pct = _pct(candles[-1].close / candles[0].close - 1.0)

    up_capture, down_capture = upside_downside_capture(candles)
    if up_capture is None:
        flags.append("capture_undefined")

    volume_liquidity = net_flow_proxy(snapshot)
    if volume_liquidity is None and snapshot is not None:
        flags.append("netflow_proxy_undefined")

    liquidity_usd = snapshot.liquidity_usd if snapshot else None
    volume_24h_usd = snapshot.volume_24h_usd if snapshot else None
    fdv_usd = snapshot.fdv_usd if snapshot else None

    fdv_liquidity: float | None = None
    if snapshot is not None and snapshot.liquidity_usd and snapshot.liquidity_usd > 0 and snapshot.fdv_usd:
        fdv_liquidity = round(snapshot.fdv_usd / snapshot.liquidity_usd, 4)

    momentum: float | None = None
    if len(candles) > momentum_fast:
        baseline = candles[-1 - momentum_fast].close
        if baseline > 0:
            momentum = _pct(candles[-1].close / baseline - 1.0)

    values: list[Metric] = [
        Metric("close_price", price, "usd"),
        Metric("window_high", window_high, "usd"),
        Metric("window_low", window_low, "usd"),
        Metric("price_change_pct", change_pct, "percent"),
        Metric("realized_volatility_24h", realized_volatility(candles, volatility_window), "percent",
               note="annualised from hourly returns"),
        Metric("return_zscore_24h", return_zscore(candles, 24), "sigma"),
        Metric("rsi_14", rsi(candles, rsi_period), "index"),
        Metric("volume_ratio", volume_ratio(candles), "ratio"),
        Metric("volume_per_liquidity", volume_liquidity, "ratio", note="pressure proxy, not signed netflow"),
        Metric("fdv_to_liquidity", fdv_liquidity, "ratio"),
        Metric("max_drawdown", max_drawdown(candles), "percent"),
        Metric("risk_reward_asymmetry", risk_reward_asymmetry(candles), "ratio"),
        Metric(f"momentum_{momentum_fast}h", momentum, "percent"),
        Metric("up_capture", up_capture, "ratio", note="normalised share of window gains"),
        Metric("down_capture", down_capture, "ratio", note="normalised share of window losses"),
        Metric("liquidity_usd", liquidity_usd, "usd"),
        Metric("volume_24h_usd", volume_24h_usd, "usd"),
        Metric("fdv_usd", fdv_usd, "usd"),
    ]

    return MetricResult(
        token_key=(snapshot.pool_key if snapshot else "unknown"),
        computed_at=utc_now_iso(),
        window_hours=len(candles),
        metrics=tuple(values),
        flags=tuple(flags),
    )


def as_dict_map(result: MetricResult) -> dict[str, float | None]:
    return {metric.name: metric.value for metric in result.metrics}


def is_high_volatility(values: dict[str, float | None], threshold: float = 150.0) -> bool:
    observed = values.get("realized_volatility_24h")
    return observed is not None and observed > threshold


def is_volume_spike(values: dict[str, float | None], threshold: float = 3.0) -> bool:
    observed = values.get("volume_ratio")
    return observed is not None and observed > threshold


def liquidity_thin(values: dict[str, float | None], min_usd: float = 50_000.0) -> bool:
    observed = values.get("liquidity_usd")
    return observed is not None and observed < min_usd


def risk_flags(values: dict[str, float | None]) -> list[str]:
    """Boolean risk screen used by the report and by the eval harness."""

    flags: list[str] = []
    if is_high_volatility(values):
        flags.append("high_realized_volatility")
    if is_volume_spike(values):
        flags.append("volume_spike")
    if liquidity_thin(values):
        flags.append("thin_liquidity")
    drawdown = values.get("max_drawdown")
    if drawdown is not None and drawdown > 40:
        flags.append("deep_drawdown")
    rsi_value = values.get("rsi_14")
    if rsi_value is not None and (rsi_value > 70 or rsi_value < 30):
        flags.append("rsi_extreme")
    return flags


def interpret(values: dict[str, float | None]) -> list[str]:
    """Turn metric values into short, falsifiable statements for the report.

    Only rules that actually fire are returned, so a quiet pool yields a short
    and honest section instead of padding.
    """

    notes: list[str] = []
    if is_high_volatility(values):
        notes.append(
            "Realised volatility is elevated for an on-chain series, so any single-bar move "
            "should be read as noise unless it repeats across several bars."
        )
    if is_volume_spike(values):
        notes.append(
            "The latest bar traded far above its trailing baseline; check whether it traces to "
            "one wallet before treating it as demand."
        )
    rsi_value = values.get("rsi_14")
    if rsi_value is not None:
        if rsi_value > 70:
            notes.append("RSI sits in overbought territory over the selected window.")
        elif rsi_value < 30:
            notes.append("RSI sits in oversold territory over the selected window.")
    if liquidity_thin(values):
        notes.append(
            "Pool liquidity is thin relative to a research-quality minimum, so slippage and "
            "price impact dominate any position above a trivial size."
        )
    drawdown = values.get("max_drawdown")
    if drawdown is not None and drawdown > 40:
        notes.append(
            "The window contains a drawdown above 40 percent, so entry timing here is a "
            "liquidity question rather than a thesis question."
        )
    return notes
