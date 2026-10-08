---
title: Net flow, whale concentration and what price data cannot show
kind: methodology
---

# Net flow, wallet concentration and the limits of price data

## The difference between flow and pressure

Signed net flow is the difference between inflows and outflows across a set of tracked
wallets. It requires labelled addresses, which the free data tier does not expose.
Without labels, the honest alternative is a pressure proxy: volume per unit of
liquidity, which says how hard the pool is being traded relative to its depth but not
in which direction.

Calling that proxy net flow would be a small lie that compounds, because a reader
would assume the sign carries meaning. The pipeline labels it a proxy for that reason.

## Wallet concentration

A pool where the top holders own most of the supply behaves differently from a
diffuse one even at identical liquidity. Concentration changes what a volume spike
means: in a concentrated pool a single transfer can dominate the tape.

Concentration cannot be derived from price and volume. When a report cannot state it,
the open questions section says so rather than dropping the topic, because a reader
who silently assumes diffuse distribution will size positions wrongly.

## What a volume spike usually is

Volume expansion has three common causes: genuine new demand, one large holder
trading, or a listing that induces arbitrage. Price data alone cannot separate these.
A ratio of latest volume to its trailing mean flags the event, and wallet level data
is what would resolve it.

For that reason a volume spike is reported as a question to investigate rather than as
a finding. Treating it as a finding is the most common way a research read turns out
to be a description of one address.

