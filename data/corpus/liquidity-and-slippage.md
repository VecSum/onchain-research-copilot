---
title: Liquidity and price impact on AMM pools
kind: methodology
---

# Liquidity and price impact on AMM pools

## What liquidity actually measures

Liquidity is the reserve depth available in a pool at the current price band. A pool
with large liquidity absorbs a given trade with a proportionally smaller price move.
This is why volume divided by liquidity is a useful first screen: a pool doing
volume far above its own depth is being pushed around by individual trades rather
than by sustained demand, and the resulting price move should not be read as
conviction.

Thin liquidity is the single most common reason a research read is wrong. When
depth is small, a single trade moves the price several percent, the printed
open-high-low-close range stops reflecting anything about the market, and every
derived metric inherits that noise.

## Slippage as the honest uncertainty

For any position above a trivial size, the relevant number is not the mid price but
the fill price after slippage. A pool with shallow depth can print an attractive
quote and still be untradeable at size. Reports that quote only the mid price hide
this, so every report in this pipeline states depth alongside the price.

## Practical thresholds

For research purposes, depth below roughly fifty thousand US dollars is treated as
thin. That threshold is a screening heuristic and not a statement about any
particular pool, and it is configurable rather than hard coded.

A volume-to-liquidity ratio above roughly three on a single bar is a signal to check
whether one wallet is responsible. Concentration is the usual explanation: a
single holder trading in and out produces the same candlestick shape as a broad
change in demand, and only wallet level data distinguishes the two.
