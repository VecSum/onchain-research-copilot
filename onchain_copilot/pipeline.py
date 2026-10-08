"""Pipeline orchestration: providers -> metrics -> retrieval -> composer.

The pipeline is deliberately linear and boring. One stage produces a value,
the next stage consumes it, and each stage boundary is where a failure is
caught and recorded. There is no callback graph and no plugin registry that
runs pieces out of order, because the whole point of a research pipeline is that
you can explain afterwards which stage produced which number.

Degradation is a first-class result, not an error path: when a stage cannot
produce its output the run continues with a recorded reason, so the reader
sees the gap instead of a silent blank.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import metrics as metrics_mod
from .config import Config
from .llm import DeterministicComposer, build_composer, facts_block
from .logging_setup import get_logger, timed
from .models import Evidence, Report, ReportSection, RunTrace
from .providers import ProviderError, build_providers, first_healthy
from .render import render_html, render_json, render_markdown, write_report
from .retrieval import Document, Retriever, build_index, load_documents

log = get_logger("pipeline")


def _compose_evidence(chunks, max_evidence: int) -> list[Evidence]:
    """Bind a retrieved chunk to the claim it supports.

    The claim is the sentence nearest the highest-scoring match rather than a
    paraphrase of it, so what the report quotes and what the corpus says stay
    literal. A client checking a citation sees their own words, not mine.
    """

    evidence: list[Evidence] = []
    for chunk in chunks[:max_evidence]:
        quote = " ".join(
            line for line in chunk.text.splitlines() if not line.lstrip().startswith("#")
        )
        quote = " ".join(quote.split())
        if len(quote) > 280:
            quote = quote[:277] + "..."
        evidence.append(
            Evidence(
                claim=quote,
                doc_id=chunk.doc_id,
                doc_title=chunk.title,
                quote=quote,
                locator=chunk.chunk_id,
                score=chunk.score,
            )
        )
    return evidence


def run_research(pool_address: str, question: str, config: Config | None = None,
                mode: str | None = None, chain: str = "bsc",
                report_name: str | None = None) -> dict[str, Any]:
    """Run one report end to end and return a serialisable result dict."""

    if config is None:
        config = Config.load("config.toml")
    mode = mode or config.get("run", "data_mode")
    trace = RunTrace()

    providers = build_providers(config, mode)
    with timed(log, "resolve_pool", pool=pool_address, mode=mode):
        pool, provider_name = first_healthy(providers, "fetch_pool_snapshot", pool_address, chain)
    candles, candle_source = first_healthy(providers, "fetch_ohlcv", pool_address, chain)
    if not candles:
        trace.degrade("no_ohlcv_series")
    trace.record("data", pool_source=provider_name,
                candle_source=candle_source, candles=len(candles))

    with timed(log, "compute_metrics", candles=len(candles)):
        result = metrics_mod.compute(
            candles,
            pool,
            volatility_window=int(config.get("metrics", "volatility_window")),
            momentum_fast=int(config.get("metrics", "momentum_fast")),
            momentum_slow=int(config.get("metrics", "momentum_slow")),
            rsi_period=int(config.get("metrics", "rsi_period")),
        )

    documents = load_documents(config.get("retrieval", "corpus_dir"))
    if not documents:
        trace.degrade("empty_corpus")
    index = build_index(
        documents,
        chunk_chars=int(config.get("retrieval", "chunk_chars")),
        chunk_overlap=int(config.get("retrieval", "chunk_overlap")),
        dim=int(config.get("retrieval", "dim")),
    )
    retriever = Retriever(
        index,
        dim=int(config.get("retrieval", "dim")),
        rrf_k=int(config.get("retrieval", "rrf_k")),
        top_k=int(config.get("retrieval", "top_k")),
        rerank_enabled=bool(config.get("retrieval", "rerank_enabled")),
        refusal_threshold=float(config.get("retrieval", "refusal_threshold")),
        max_evidence=int(config.get("retrieval", "max_evidence_per_report")),
    )

    with timed(log, "retrieve", chunks=len(index.chunks)):
        hits = retriever.search(question)
    evidence = _compose_evidence(hits, int(config.get("retrieval", "max_evidence_per_report")))
    trace.record("retrieval", documents=len(documents), hits=len(hits),
                evidence=len(evidence))

    composer, composer_name = build_composer(config)
    facts = facts_block(result, evidence, question)
    degraded_composer = False
    try:
        with timed(log, "compose", composer=composer_name):
            narrative = composer.complete(facts)
            if composer_name != "deterministic" and isinstance(composer, DeterministicComposer):
                degraded_composer = True
    except Exception as exc:  # composer boundary: degrade, do not fail the run
        trace.degrade(f"composer_failed:{type(exc).__name__}")
        narrative = DeterministicComposer().complete(facts)
        composer_name = "deterministic"
        degraded_composer = True
        log.warning("composer failed, fell back to deterministic: %s", exc)

    if not evidence:
        trace.degrade("no_evidence")

    sections = _sections_from_markdown(narrative, evidence)
    token = getattr(pool, "symbol", None) or "UNKNOWN"
    flags = tuple(sorted(set(list(result.flags) + (["composer_degraded"] if degraded_composer else []))))
    report = Report(
        token=_token_like(pool, token),
        generated_at=result.computed_at,
        question=question,
        headline=_headline(pool, result),
        sections=tuple(sections),
        evidence=tuple(evidence),
        metrics=result.metrics,
        flags=flags,
        unverified=tuple(trace.degraded),
        grounded=bool(evidence),
        author=f"onchain-research-copilot/{composer_name}",
    )
    trace.record("report", grounded=report.grounded, evidence=len(evidence))

    payload = {
        "report": _payload(report),
        "_markdown": render_markdown(report, config.get("report", "disclaimer")),
        "trace": trace.to_dict(),
        "provider": provider_name,
        "candle_source": candle_source,
    }
    written = write_report(report, "output", disclaimer=config.get("report", "disclaimer"),
                         name=report_name)
    payload["artifacts"] = written
    return payload


def _token_like(pool, symbol: str):
    from .models import Token

    return Token(symbol=symbol, name=str(getattr(pool, "symbol", symbol)),
                 chain=str(getattr(pool, "chain", "bsc")),
                 address=str(getattr(pool, "pool_key", "")))
def _headline(pool, result) -> str:
    symbol = str(getattr(pool, "symbol", "UNKNOWN"))
    price = result.value("close_price")
    change = result.value("price_change_pct")
    if price is None or change is None:
        return f"{symbol}: insufficient data for a directional read"
    direction = "up" if change > 0 else ("down" if change < 0 else "flat")
    move = f"{abs(float(change)):.2f}%"
    return f"{symbol} is {direction} {move} over the loaded window"


def _sections_from_markdown(narrative: str, evidence) -> list[ReportSection]:
    """Split composer output into heading/body sections, verbatim.

    The composer owns the prose and the renderer only structures it, so there
    is one source of truth for what the report claims.
    """

    sections: list[ReportSection] = []
    current_heading = "Report"
    current_lines: list[str] = []

    def flush() -> None:
        body_text = chr(10).join(current_lines).strip()
        if body_text:
            sections.append(ReportSection(heading=current_heading, body=body_text))

    for raw in narrative.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    if not sections:
        sections.append(ReportSection(heading="Report", body=narrative.strip()))
    return sections


def _payload(report: Report) -> dict[str, Any]:
    from .render import report_to_dict

    return report_to_dict(report)


def list_tokens(config: Config | None = None, mode: str | None = None,
                chain: str = "bsc", limit: int = 20) -> list[dict[str, str]]:
    """List pools available from the configured sources, for discovery."""

    config = Config() if config is None else config
    mode = mode or config.get("run", "data_mode")
    providers = build_providers(config, mode)
    pools, source = first_healthy(providers, "list_pools", chain, limit)
    return [
        {
            "symbol": pool.base_token.symbol,
            "name": pool.base_token.name,
            "chain": pool.base_token.chain,
            "pool_address": pool.pool_address,
            "dex": pool.dex,
            "quote": pool.quote_symbol,
            "source": source,
        }
        for pool in pools
    ]


def render_only(report: Report, config: Config, output_dir: str = "output") -> dict[str, str]:
    """Re-render an already computed report. Used by tests and the demo."""

    disclaimer = config.get("report", "disclaimer")
    return {
        "markdown": render_markdown(report, disclaimer),
        "html": render_html(report, disclaimer),
        "json": render_json(report),
    }
