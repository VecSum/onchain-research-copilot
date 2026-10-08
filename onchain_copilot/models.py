"""Core value objects.

Frozen dataclasses everywhere: a metric result or a retrieved chunk is an
immutable fact about a run, never something later stages are allowed to edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Token:
    symbol: str
    name: str
    chain: str
    address: str

    @property
    def key(self) -> str:
        return f"{self.chain}:{self.address}".lower()


@dataclass(frozen=True)
class Pool:
    pool_address: str
    dex: str
    quote_symbol: str
    base_token: Token
    created_at: str | None = None

    @property
    def key(self) -> str:
        return self.pool_address.lower()


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Snapshot:
    """One observation of a pool at a point in time."""

    pool_key: str
    symbol: str
    chain: str
    ts: str
    price_usd: float | None
    liquidity_usd: float | None
    volume_24h_usd: float | None
    fdv_usd: float | None
    source: str


@dataclass(frozen=True)
class MetricSeries:
    name: str
    values: list[float]


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | None
    unit: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "note": self.note,
        }


@dataclass(frozen=True)
class MetricResult:
    token_key: str
    computed_at: str
    window_hours: int
    metrics: tuple[Metric, ...]
    flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "token_key": self.token_key,
            "computed_at": self.computed_at,
            "window_hours": self.window_hours,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "flags": list(self.flags),
        }

    def value(self, name: str) -> float | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric.value
        return None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    lexical_score: float
    semantic_score: float


@dataclass(frozen=True)
class Evidence:
    claim: str
    doc_id: str
    doc_title: str
    quote: str
    locator: str
    score: float


@dataclass(frozen=True)
class ReportSection:
    heading: str
    body: str


@dataclass(frozen=True)
class Report:
    token: Token
    generated_at: str
    question: str
    headline: str
    sections: tuple[ReportSection, ...]
    evidence: tuple[Evidence, ...]
    metrics: tuple[Metric, ...]
    flags: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    author: str = "onchain-research-copilot"
    grounded: bool = True


@dataclass
class RunTrace:
    """Aggregated timing/degradation record for one report run."""

    stages: dict[str, Any] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    def record(self, stage: str, **fields: Any) -> None:
        self.stages[stage] = fields

    def degrade(self, reason: str) -> None:
        if reason not in self.degraded:
            self.degraded.append(reason)

    def to_dict(self) -> dict[str, Any]:
        return {"stages": self.stages, "degraded": self.degraded}
