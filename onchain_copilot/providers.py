"""Market-data provider adapters.

Two shapes of provider live behind one interface: NetworkProvider talks to the
free public tier of a real data source, and FixtureProvider replays a bundled
snapshot so a run is reproducible with no network, no API key and no rate
limit. Adding a provider means adding one class and one entry in PROVIDERS.

Providers never raise past this module's public helpers: first_healthy() walks
the configured order and skips anything that fails, so the run degrades to the
next source instead of dying.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Candle, Pool, Snapshot, Token, utc_now_iso


class ProviderError(RuntimeError):
    """A provider could not produce usable data."""


def build_headers(user_agent: str) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept": "application/json"}


def http_get_json(url: str, headers: dict[str, str], timeout_seconds: float, retries: int = 2) -> Any:
    """GET with bounded retries and idle timeout. Raises ProviderError at the end."""

    last_error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt * 0.25, 1.5))
    raise ProviderError(f"{url} failed after {retries + 1} attempts: {last_error}")


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class GeckoTerminalProvider:
    """GeckoTerminal free public tier.

    The free tier allows roughly 30 calls per minute, which is enough for a few
    pools per report and nowhere near enough for a sweep over thousands.
    """

    name = "geckoterminal"
    base = "https://api.geckoterminal.com/api/v2"

    def __init__(self, timeout_seconds: float = 12, retries: int = 2, user_agent: str = "onchain-research-copilot/0.1") -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.headers = build_headers(user_agent)

    def list_pools(self, chain: str, limit: int) -> list[Pool]:
        url = f"{self.base}/networks/{chain}/trending_pools"
        payload = http_get_json(url, self.headers, self.timeout_seconds, self.retries)
        return self._parse_pools(payload)[:limit]

    def fetch_pool_snapshot(self, pool_address: str, chain: str) -> Snapshot:
        url = f"{self.base}/networks/{chain}/pools/{pool_address}"
        payload = http_get_json(url, self.headers, self.timeout_seconds, self.retries)
        data = payload.get("data") or {}
        attrs = data.get("attributes") or {}
        base_href = (data.get("relationships") or {}).get("base_token", {}).get("data", {}).get("id", "")
        symbol = base_href.rsplit("_", 1)[-1] if "_" in base_href else (attrs.get("name") or "?")
        reserve = attrs.get("reserve_in_usd")
        liquidity = reserve.get("value") if isinstance(reserve, dict) else reserve
        volume = attrs.get("volume_usd") or {}
        return Snapshot(
            pool_key=f"{chain}:{pool_address}".lower(),
            symbol=symbol,
            chain=chain,
            ts=attrs.get("pool_created_at") or utc_now_iso(),
            price_usd=_number(attrs.get("base_token_price_usd")),
            liquidity_usd=_number(liquidity),
            volume_24h_usd=_number(volume.get("h24") if isinstance(volume, dict) else volume),
            fdv_usd=_number(attrs.get("fdv_usd")),
            source=self.name,
        )

    def fetch_ohlcv(self, pool_address: str, chain: str, interval: str = "1h", limit: int = 168) -> list[Candle]:
        url = f"{self.base}/networks/{chain}/pools/{pool_address}/ohlcv/{interval}"
        payload = http_get_json(
            f"{url}?aggregate=1&limit={int(limit)}", self.headers, self.timeout_seconds, self.retries
        )
        rows = (payload.get("data") or {}).get("attributes") or {}
        candles: list[Candle] = []
        for row in rows.get("ohlcv_list") or []:
            if len(row) < 6:
                continue
            candles.append(
                Candle(
                    ts=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        candles.sort(key=lambda candle: candle.ts)
        return candles

    def _parse_pools(self, payload: dict[str, Any]) -> list[Pool]:
        pools: list[Pool] = []
        for item in payload.get("data") or []:
            attrs = item.get("attributes") or {}
            relationships = item.get("relationships") or {}
            base = (relationships.get("base_token") or {}).get("data") or {}
            quote = (relationships.get("quote_token") or {}).get("data") or {}
            token_ref = base.get("id") or ""
            address = token_ref.rsplit("_", 1)[-1] if "_" in token_ref else ""
            quote_ref = quote.get("id") or ""
            pools.append(
                Pool(
                    pool_address=attrs.get("address") or "",
                    dex=attrs.get("dex_id") or self.name,
                    quote_symbol=quote_ref.rsplit("_", 1)[-1] if "_" in quote_ref else "",
                    base_token=Token(
                        symbol=address or (attrs.get("name") or "?"),
                        name=attrs.get("name") or "",
                        chain=self.name,
                        address=address,
                    ),
                    created_at=attrs.get("pool_created_at"),
                )
            )
        return [pool for pool in pools if pool.pool_address]


class DexScreenerProvider:
    """DexScreener free public tier: single-pair reads plus a keyword search."""

    name = "dexscreener"
    base = "https://api.dexscreener.com"

    def __init__(self, timeout_seconds: float = 12, retries: int = 2, user_agent: str = "onchain-research-copilot/0.1") -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.headers = build_headers(user_agent)

    def list_pools(self, chain: str, limit: int) -> list[Pool]:
        payload = http_get_json(
            f"{self.base}/latest/dex/search?q={urllib.parse.quote(chain)}",
            self.headers,
            self.timeout_seconds,
            self.retries,
        )
        pools = [self._parse_pair(pair) for pair in payload.get("pairs") or []]
        return [pool for pool in pools if pool.pool_address][:limit]

    def fetch_pool_snapshot(self, pool_address: str, chain: str) -> Snapshot:
        payload = http_get_json(
            f"{self.base}/latest/dex/pairs/{chain}/{pool_address}",
            self.headers,
            self.timeout_seconds,
            self.retries,
        )
        pairs = payload.get("pairs") or []
        if not pairs:
            raise ProviderError("dexscreener returned no pair data")
        pair = pairs[0]
        return Snapshot(
            pool_key=str(pair.get("chainId") or chain).lower() + ":" + str(pair.get("pairAddress") or ""),
            symbol=(pair.get("baseToken") or {}).get("symbol") or "",
            chain=chain,
            ts=utc_now_iso(),
            price_usd=_number(pair.get("priceUsd")),
            liquidity_usd=_number((pair.get("liquidity") or {}).get("usd")),
            volume_24h_usd=_number((pair.get("volume") or {}).get("h24")),
            fdv_usd=_number(pair.get("fdv")),
            source=self.name,
        )

    def fetch_ohlcv(self, pool_address: str, chain: str, interval: str = "1h", limit: int = 168) -> list[Candle]:
        raise ProviderError("dexscreener free tier exposes no OHLCV endpoint; use geckoterminal")

    def _parse_pair(self, pair: dict[str, Any]) -> Pool:
        base = pair.get("baseToken") or {}
        return Pool(
            pool_address=pair.get("pairAddress") or "",
            dex=pair.get("dexId") or self.name,
            quote_symbol=(pair.get("quoteToken") or {}).get("symbol") or "",
            base_token=Token(
                symbol=base.get("symbol") or "",
                name=base.get("name") or "",
                chain=str(pair.get("chainId") or "").lower(),
                address=base.get("address") or "",
            ),
            created_at=None,
        )


def synthetic_pool(seed: str, hours: int = 168, chain: str = "synthetic") -> dict[str, Any]:
    """Deterministic pseudo-series for smoke tests: same seed, same numbers."""

    rng = random.Random(f"{chain}:{seed}")
    price = 1.0 + rng.random() * 4.0
    candles: list[list[float]] = []
    start = int(datetime.now(timezone.utc).timestamp()) - hours * 3600
    for hour in range(hours):
        drift = (rng.random() - 0.48) * 0.06
        open_price = price
        close_price = max(0.0001, price * (1 + drift))
        high = max(open_price, close_price) * (1 + rng.random() * 0.02)
        low = min(open_price, close_price) * (1 - rng.random() * 0.02)
        candles.append(
            [
                start + hour * 3600,
                round(open_price, 8),
                round(high, 8),
                round(low, 8),
                round(close_price, 8),
                round(50_000 + rng.random() * 900_000, 2),
            ]
        )
        price = close_price
    address = "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40))
    return {
        "chain": chain,
        "symbol": seed.upper(),
        "name": f"{seed.upper()} synthetic pool",
        "pool_address": address,
        "token_address": address,
        "quote_symbol": "USDT",
        "dex": "synthetic",
        "created_at": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(timespec="seconds"),
        "price_usd": round(price, 8),
        "liquidity_usd": round(250_000 + rng.random() * 2_000_000, 2),
        "volume_24h_usd": round(1_000_000 + rng.random() * 9_000_000, 2),
        "fdv_usd": round(5_000_000 + rng.random() * 90_000_000, 2),
        "ohlcv": candles,
    }


PROVIDERS: dict[str, type] = {
    GeckoTerminalProvider.name: GeckoTerminalProvider,
    DexScreenerProvider.name: DexScreenerProvider,
}


def fixture_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "fixtures" / "pools.json"


class FixtureProvider:
    """Replays a bundled JSON snapshot. Default for tests, demos and CI."""

    name = "fixtures"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else fixture_path()
        if not self.path.exists():
            raise ProviderError(f"fixture file not found: {self.path}")
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))

    def _entries(self, chain: str) -> list[dict[str, Any]]:
        rows = [entry for entry in self.payload.get("pools", []) if entry.get("pool_address")]
        if chain:
            matched = [entry for entry in rows if str(entry.get("chain", "")).lower() == chain.lower()]
            return matched or rows
        return rows

    def _find(self, pool_address: str) -> dict[str, Any]:
        for entry in self.payload.get("pools", []):
            if str(entry.get("pool_address", "")).lower() == pool_address.lower():
                return entry
        raise ProviderError(f"no fixture for pool {pool_address}")

    def list_pools(self, chain: str, limit: int) -> list[Pool]:
        return [self._pool_from(entry) for entry in self._entries(chain)][:limit]

    def fetch_pool_snapshot(self, pool_address: str, chain: str) -> Snapshot:
        entry = self._find(pool_address)
        return Snapshot(
            pool_key=f"{entry.get('chain', chain)}:{entry.get('pool_address')}".lower(),
            symbol=entry.get("symbol", "?"),
            chain=entry.get("chain", chain),
            ts=entry.get("ts") or utc_now_iso(),
            price_usd=_number(entry.get("price_usd")),
            liquidity_usd=_number(entry.get("liquidity_usd")),
            volume_24h_usd=_number(entry.get("volume_24h_usd")),
            fdv_usd=_number(entry.get("fdv_usd")),
            source=self.name,
        )

    def fetch_ohlcv(self, pool_address: str, chain: str, interval: str = "1h", limit: int = 168) -> list[Candle]:
        entry = self._find(pool_address)
        candles = [Candle(*row) for row in entry.get("ohlcv", [])]  # type: ignore[misc]
        candles.sort(key=lambda candle: candle.ts)
        return candles[-int(limit):]

    @staticmethod
    def _pool_from(entry: dict[str, Any]) -> Pool:
        return Pool(
            pool_address=entry.get("pool_address", ""),
            dex=entry.get("dex", "fixture"),
            quote_symbol=entry.get("quote_symbol", "USDT"),
            base_token=Token(
                symbol=entry.get("symbol", "?"),
                name=entry.get("name", entry.get("symbol", "?")),
                chain=entry.get("chain", "bsc"),
                address=entry.get("token_address", entry.get("pool_address", "")),
            ),
            created_at=entry.get("created_at"),
        )


class SyntheticProvider:
    """Deterministic generated series. Useful when no fixture matches a symbol."""

    name = "synthetic"

    @staticmethod
    def _address(seed: str) -> str:
        import hashlib

        return "0x" + hashlib.sha256(seed.encode()).hexdigest()[:40]

    def list_pools(self, chain: str, limit: int) -> list[Pool]:
        entry = synthetic_pool("demo", chain=chain)
        return [FixtureProvider._pool_from(entry)][:limit]

    def fetch_pool_snapshot(self, pool_address: str, chain: str) -> Snapshot:
        entry = synthetic_pool("demo", chain=chain)
        return Snapshot(
            pool_key=f"{chain}:{pool_address}".lower(),
            symbol=entry["symbol"],
            chain=chain,
            ts=entry["created_at"],
            price_usd=entry["price_usd"],
            liquidity_usd=entry["liquidity_usd"],
            volume_24h_usd=entry["volume_24h_usd"],
            fdv_usd=entry["fdv_usd"],
            source=self.name,
        )

    def fetch_ohlcv(self, pool_address: str, chain: str, interval: str = "1h", limit: int = 168) -> list[Candle]:
        entry = synthetic_pool("demo", chain=chain)
        return [Candle(*row) for row in entry["ohlcv"]][-int(limit):]  # type: ignore[misc]


def build_providers(config, mode: str) -> list:
    """Resolve configured provider order for a run."""

    if mode == "fixtures":
        return [FixtureProvider()]
    if mode == "synthetic":
        return [SyntheticProvider()]
    providers: list = []
    for name in config.section("providers").get("order", []):
        cls = PROVIDERS.get(name)
        if cls is None:
            continue
        providers.append(
            cls(
                timeout_seconds=config.get("providers", "timeout_seconds"),
                retries=config.get("providers", "retries"),
                user_agent=config.get("providers", "user_agent"),
            )
        )
    if not providers:
        raise ProviderError("no providers configured")
    return providers


def first_healthy(providers: Iterable[Any], method: str, *args: Any, **kwargs: Any) -> tuple[Any, str]:
    """Try each provider in order and return (result, provider_name).

    A failing provider is skipped, not fatal: the run falls to the next source
    and the caller records which one answered so the report can cite it.
    """

    errors: list[str] = []
    for provider in providers:
        try:
            return getattr(provider, method)(*args, **kwargs), provider.name
        except Exception as exc:  # provider boundary: degrade, never crash the run
            errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
    raise ProviderError("all providers failed -> " + "; ".join(errors))
