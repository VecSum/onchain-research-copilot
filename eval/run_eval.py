"""Offline evaluation harness.

Measures the two properties this pipeline claims to have, and refuses to
measure what it cannot. Recall@k: does the retriever surface the document
that answers the question. Refusal accuracy: when the corpus has nothing
to say, does the report say so instead of citing the nearest neighbour.
Faithfulness: every number in the report traces back to a computed metric.

Every case is scored on its own, so a regression names the case that broke
instead of hiding inside an average. Write the report with --report.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from onchain_copilot.config import Config, ConfigError, load_config  # noqa: E402
from onchain_copilot.pipeline import run_research  # noqa: E402
from onchain_copilot.providers import FixtureProvider  # noqa: E402
from onchain_copilot.retrieval import (  # noqa: E402
    Retriever,
    build_index,
    load_documents,
)

DEFAULT_POOL = "0xdemo000000000000000000000000000000000001"
METRIC_NAMES = {
    "close_price", "price_change_pct", "realized_volatility_24h",
    "max_drawdown", "rsi_14", "volume_ratio", "volume_per_liquidity",
    "fdv_to_liquidity", "liquidity_usd", "volume_24h_usd", "fdv_usd",
    "window_high", "window_low", "return_zscore_24h",
    "risk_reward_asymmetry", "up_capture", "down_capture", "momentum_7h",
}


def load_dataset(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def build_retriever(config: Config) -> Retriever:
    """Reuse the pipeline's own retriever so the eval and the app agree."""

    documents = load_documents(config.get("retrieval", "corpus_dir"))
    index = build_index(
        documents,
        chunk_chars=int(config.get("retrieval", "chunk_chars")),
        chunk_overlap=int(config.get("retrieval", "chunk_overlap")),
        dim=int(config.get("retrieval", "dim")),
    )

    return Retriever(
        index,
        dim=int(config.get("retrieval", "dim")),
        rrf_k=int(config.get("retrieval", "rrf_k")),
        top_k=int(config.get("retrieval", "top_k")),
        rerank_enabled=bool(config.get("retrieval", "rerank_enabled")),
        refusal_threshold=float(config.get("retrieval", "refusal_threshold")),
        max_evidence=int(config.get("retrieval", "max_evidence_per_report")),
    )


def score_case(case: dict, retriever: Retriever, k: int) -> dict:
    question = case["question"]
    expected = set(case.get("expected_docs", []))
    answerable = bool(case.get("answerable", True))
    hits = retriever.search(question)
    retrieved = [hit.doc_id for hit in hits]
    hit_ranks = [rank for rank, doc_id in enumerate(retrieved, start=1)
                  if doc_id in expected]
    return {
        "question": question,
        "answerable": answerable,
        "retrieved": retrieved,
        "expected": sorted(expected),
        "rank": hit_ranks[0] if hit_ranks else None,
        "recall_at_k": 1.0 if any(rank <= k for rank in hit_ranks) else 0.0,
        "refusal_correct": answerable == bool(retrieved),
    }


def ratio(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r[key]) / len(rows), 4)


def write_report(path: Path, retrieval_rows, pipeline_rows, k, pool, duration_s) -> None:
    answerable = [r for r in retrieval_rows if r["answerable"]]
    unanswerable = [r for r in retrieval_rows if not r["answerable"]]
    recall_at_k = ratio(answerable, "recall_at_k")
    refusal = ratio(unanswerable, "refusal_correct")
    grounded = ratio(pipeline_rows, "grounded")
    faithful = ratio(pipeline_rows, "faithful")
    latencies = [r["elapsed_ms"] for r in pipeline_rows] or [0.0]
    median_latency = round(statistics.median(latencies), 1)
    p95 = round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)
    lines = [
        "# Evaluation report",
        "",
        "Pool: `" + pool + "`  |  cases: " + str(len(retrieval_rows))
        + "  |  harness runtime: " + format(duration_s, ".1f") + "s",
        "Retrieval Recall@" + str(k) + ": **" + format(recall_at_k, ".0%")
        + "** of answerable cases surfaced an expected document",
        "Refusal accuracy: **" + format(refusal, ".0%")
        + "** of off-corpus cases returned no evidence",
        "Grounded rate: **" + format(grounded, ".0%")
        + "** of end-to-end runs produced a non-empty evidence set",
        "Faithfulness: **" + format(faithful, ".0%")
        + "** of reports quote only computed metric values",
        "",
        "Latency is one cold run of the whole pipeline on one machine, so it is a",
        "smoke number rather than a benchmark.",
        "",
        "- median end-to-end: " + str(median_latency) + " ms",
        "- p95 end-to-end: " + str(p95) + " ms",
        "",
        "## Retrieval, per case",
        "",
        "| question | expected | retrieved | hit@k | refusal correct |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in retrieval_rows:
        got = ", ".join(row["retrieved"]) or "-"
        exp = ", ".join(row["expected"]) or "-"
        flag = "yes" if row["refusal_correct"] else "**NO**"
        hit = format(row["recall_at_k"], ".0%")
        lines.append(
            "| " + row["question"] + " | " + exp + " | " + got
            + " | " + hit + " | " + flag + " |"
        )
    lines += [
        "",
        "## End-to-end, per case",
        "",
        "| question | grounded | faithful | evidence docs | refused with reason | latency ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in pipeline_rows:
        docs = ", ".join(row["evidence_docs"]) or "-"
        lines.append(
            "| " + row["question"] + " | " + str(row["grounded"])
            + " | " + str(row["faithful"]) + " | " + docs
            + " | " + str(row["refused_with_reason"])
            + " | " + str(row["elapsed_ms"]) + " |"
        )
    lines += [
        "",
        "## How these are measured",
        "",
        "Recall@k counts a case as hit when any expected document appears in the top",
        str(k) + " retrieved chunks. Refusal correctness requires the retriever to",
        "return nothing for an off-corpus question and something for an on-corpus",
        "one; the failure mode it guards against is a confident citation of the",
        "nearest neighbour. Faithfulness re-parses the numbers printed in each",
        "report section and requires every one to equal a value the metrics stage",
        "computed. The tokens 0, 1, 14, 24 and 40 are exempt because they appear in",
        "fixed labels such as RSI(14).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval", description="offline pipeline eval")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--report", default="output/eval-report.md")
    parser.add_argument("--k", type=int, default=3, help="recall cut-off")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="retrieval metrics only, no end-to-end runs")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    cases = load_dataset(Path(args.dataset))
    if not cases:
        print(f"no cases in {args.dataset}", file=sys.stderr)
        return 1

    retriever = build_retriever(config)
    retrieval_rows = [score_case(case, retriever, args.k) for case in cases]

    pipeline_rows = []
    started = time.perf_counter()
    if not args.skip_pipeline:
        import logging

        # The pipeline logs each stage to stderr; keep the eval output readable.
        logging.disable(logging.INFO)
        try:
            for case in cases:
                pipeline_rows.append(score_pipeline(case, config, args.pool))
        finally:
            logging.disable(logging.NOTSET)
    duration = time.perf_counter() - started

    write_report(Path(args.report), retrieval_rows, pipeline_rows,
                 args.k, args.pool, duration)
    print(f"wrote {args.report}")
    return 0


def faithfulness(report: dict) -> bool:
    """Every number in composer prose must trace back to a computed metric.

    This is the check that stops a composer from inventing a figure. It is
    deliberately blunt: extract every numeric token from the report prose and
    require it to match a computed metric value, allowing for rounding and for
    fixed labels such as RSI(14) or list markers.

    Numbers inside a verbatim evidence quote are exempt, because those words
    came from the corpus and the composer cannot edit them. Without that
    exemption a quoting document that mentions a small integer would read as a
    fabricated figure instead of a citation.
    """

    values = [
        float(item["value"])
        for item in report["metrics"]
        if item.get("value") is not None
    ]
    quoted = " ".join(str(item.get("quote", "")) for item in report.get("evidence", []))
    cited = set(re.findall(r"-?[0-9]+(?:[.][0-9]+)?", quoted))
    for section in report["sections"]:
        # Strip the [eN] citation markers first: they are indices, not claims
        # about the world, so a "[e3]" must not read as a fabricated figure.
        prose = re.sub(r"\[e\d+\]", " ", section["body"])
        for token in re.findall(r"-?[0-9]+(?:[.][0-9]+)?", prose):
            if token in cited:
                continue
            try:
                number = float(token)
            except ValueError:
                continue
            if any(abs(number - value) < 1e-6 for value in values):
                continue
            decimals = len(token.split(".")[1]) if "." in token else 0
            if any(round(value, decimals) == number for value in values):
                continue
            if number == int(number) and int(number) in {0, 1, 14, 24, 40}:
                continue  # fixed labels such as RSI(14), list markers
            return False
    return True


def score_pipeline(case: dict, config: Config, pool: str) -> dict:
    """Run one report end to end and check what the report claims."""

    started = time.perf_counter()
    outcome = run_research(pool, case["question"], config=config, mode="fixtures")
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    report = outcome["report"]  # serialised form, same numbers the Report holds
    return {
        "question": case["question"],
        "grounded": report["grounded"],
        "answerable": bool(case.get("answerable", True)),
        "evidence_docs": [item["doc_id"] for item in report["evidence"]],
        "refused_with_reason": (not report["grounded"]) and bool(outcome["trace"]["degraded"]),
        "faithful": faithfulness(report),
        "elapsed_ms": elapsed_ms,


        "expected_docs": sorted(set(case.get("expected_docs", []))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
