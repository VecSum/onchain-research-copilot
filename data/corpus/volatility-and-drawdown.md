---
title: Volatility, drawdown and position sizing
kind: methodology
---

# Volatility, drawdown and position sizing

## Annualised volatility from hourly candles

Realised volatility is estimated by taking the standard deviation of hourly simple
returns over a trailing window and scaling it by the square root of the number of
hours in a year. The scaling factor is 24 times 365, which assumes the series is
genuinely hourly and has no gaps.

A series with fewer than three valid returns produces no estimate at all. Reporting a
volatility number computed from two observations would be worse than reporting
nothing, because the reader would treat it as a measurement.

## Why drawdown matters more than total return

Two series can end the window at the same price with very different histories. The
one that fell forty percent on the way is materially harder to hold, and it changes
what position size is defensible. Maximum drawdown captures the worst peak to trough
decline and is the single most useful risk statistic over a short window.

A drawdown above forty percent inside a week is a liquidity story before it is a
thesis story: something in the pool structure changed, and the candlestick shape does
not explain by itself what.

## Risk reward asymmetry

Sharpe-like ratios compress long tailed on-chain series into a single number that
looks tidy and says little. The mean of up-bars divided by the mean of down-bars keeps
the shape visible: a pool whose average gain exceeds its average loss is easier to
size than one with the opposite profile, even when both have the same total return.

## Volatility and window choice

Volatility is only comparable across the same window length and the same cadence. A
24 hour window on hourly candles and a 24 hour window on daily candles are different
numbers, and quoting one against the other is a common source of confusion.

When volatility is elevated, single bar moves should be treated as noise unless they
repeat across several bars. Repetition is what distinguishes a regime change from a
liquidation cascade.

