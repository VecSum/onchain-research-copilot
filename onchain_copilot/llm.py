"""LLM adapter.

The report composer is a pluggable stage. The default is deterministic: the
prose is assembled from templates against the computed metrics and the cited
evidence, so a run produces byte-identical output and can be committed as a
golden. That is what makes the eval harness reproducible offline.

Switching to a real model changes one config key. Any OpenAI-compatible Chat
Completions endpoint works, which covers local vLLM/MLX deployments and hosted
APIs alike. The prompt is built from the same structured inputs, so the model
sees exactly the facts the deterministic path would have quoted - it never sees
raw market data and is never asked to do arithmetic.

Secrets are read from an environment variable named by the config file. The
config file itself never holds a key, which keeps them out of version control
by construction rather than by discipline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Sequence

from .models import Chunk, Evidence, Metric, MetricResult


class LLMError(RuntimeError):
    """Raised when the configured LLM cannot produce a usable answer."""


def facts_block(metrics: MetricResult, evidence: Sequence[Evidence], question: str) -> dict[str, Any]:
    """The single structured payload both composers consume.

    Metrics are rounded before they reach the prompt: handing a model full
    float precision invites it to do arithmetic it is bad at, and the report
    is supposed to quote numbers this pipeline computed, not numbers the model
    reconstructed.
    """

    return {
        "question": question,
        "metrics": [
            {"name": metric.name, "value": metric.value, "unit": metric.unit,
             "note": metric.note}
            for metric in metrics.metrics
        ],
        "flags": list(metrics.flags),
        "evidence": [
            {"doc_id": item.doc_id, "doc_title": item.doc_title,
             "quote": item.quote, "score": item.score}
            for item in evidence
        ],
    }


SYSTEM_PROMPT = """You are the report composer for an on-chain research pipeline.
You are given computed metrics and retrieved evidence as JSON.

Rules:
- Quote numbers exactly as given. Never compute a new number.
- Every factual claim about the project must cite one evidence id as [e1], [e2].
- If the evidence list is empty, say the corpus contains no evidence for the
  question and stop. Do not speculate, do not fill gaps from general knowledge.
- Never give a buy, sell or price-target instruction.
- Output plain Markdown only: no preamble, no code fences.
"""


class DeterministicComposer:
    """Template composer. Same inputs, byte-identical output. No network."""

    name = "deterministic"

    def complete(self, facts: dict[str, Any]) -> str:
        return render_template(facts)


class OpenAICompatComposer:
    """Any Chat Completions endpoint. Key comes from the environment."""

    name = "openai_compat"

    def __init__(self, base_url: str, model: str, api_key: str | None,
                 temperature: float = 0.1, max_tokens: int = 1200,
                 timeout_seconds: float = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, facts: dict[str, Any]) -> str:
        if not self.api_key:
            raise LLMError("no API key in the configured environment variable")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(facts, indent=2, sort_keys=True)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "onchain-research-copilot/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise LLMError(f"{self.base_url} did not return a completion: {exc}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise LLMError("completion response contained no choices")
        content = (choices[0].get("message") or {}).get("content")
        if not content or not str(content).strip():
            raise LLMError("completion response contained no content")
        return str(content).strip()


def render_template(facts: dict[str, Any]) -> str:
    """Assemble the Markdown narrative from structured facts only.

    Because this never sees raw prices, it cannot invent a direction: every
    sentence is a conditional over a metric value computed upstream. When there
    is no evidence it refuses, in the same sentence every time.
    """

    metrics = {item["name"]: item for item in facts.get("metrics", [])}
    evidence = facts.get("evidence", [])
    flags = facts.get("flags", [])

    def value(name: str):
        item = metrics.get(name)
        return item["value"] if item else None

    def num(name: str, spec: str) -> str:
        observed = value(name)
        return "n/a" if observed is None else format(float(observed), spec)

    def pct(name: str) -> str:
        observed = value(name)
        return "n/a" if observed is None else f"{float(observed):+.2f}%"

    lines: list[str] = []
    lines.append("## Verdict")
    if not evidence:
        lines.append("The corpus contains no evidence addressing this question. The"
                     " metrics below are computed from price and liquidity data"
                     " alone, and no project claim is asserted.")
    else:
        lines.append("Metrics are computed and reproducible. Every project claim"
                     " below is supported by the cited evidence and nothing else.")
    lines.append("")

    lines.append("## Measured behaviour")
    lines.append(f"- Close: {num('close_price', '.8g')} USD")
    lines.append(f"- Window change: {pct('price_change_pct')}")
    lines.append(f"- Realised volatility (annualised): {pct('realized_volatility_24h')}")
    lines.append(f"- Max drawdown: {num('max_drawdown', '.2f')} pct")
    lines.append(f"- RSI(14): {num('rsi_14', '.1f')}")
    lines.append(f"- Volume ratio (latest / trailing mean): {num('volume_ratio', '.2f')}x")
    lines.append(f"- Volume per liquidity dollar (pressure proxy): {num('volume_per_liquidity', '.2f')}x")
    lines.append(f"- FDV / liquidity: {num('fdv_to_liquidity', '.2f')}x")
    lines.append("")
    if flags:
        lines.append("## Data caveats")
        for flag in flags:
            lines.append(f"- {flag}")
        lines.append("")

    lines.append("## What the corpus says")
    if evidence:
        for index, item in enumerate(evidence, start=1):
            citation = item.get("doc_title") or item.get("doc_id")
            quote = " ".join(str(item.get("quote", "")).split())
            if len(quote) > 240:
                quote = quote[:237] + "..."
            lines.append(f"- [e{index}] {quote} ({citation})")
    else:
        lines.append("- No document in the corpus addresses this question.")
    lines.append("")

    lines.append("## Open questions")
    lines.append("- Wallet-level net flow and holder concentration are not available"
                 " from the configured sources; both would change the read above.")
    lines.append("- Whether the latest volume expansion traces to one wallet or many"
                 " is not resolvable from price data alone.")
    lines.append("")
    lines.append("## Not investment advice")
    lines.append("This is a research artefact, not a recommendation to transact.")
    return "\n".join(lines)


def build_composer(config):
    """Instantiate the configured composer, falling back when it cannot run.

    The fallback is not a convenience feature. A missing key or an unreachable
    endpoint should degrade the run to deterministic mode and be reported, not
    turn a research request into a stack trace.
    """

    section = config.section("llm")
    provider = section.get("provider", "deterministic")
    if provider == "deterministic":
        return DeterministicComposer(), provider
    api_key = os.environ.get(section.get("api_key_env", "LLM_API_KEY")) or None
    composer = OpenAICompatComposer(
        base_url=section.get("base_url", "http://127.0.0.1:8080/v1"),
        model=section.get("model", "qwen2.5-7b-instruct"),
        api_key=api_key,
        temperature=float(section.get("temperature", 0.1)),
        max_tokens=int(section.get("max_tokens", 1200)),
        timeout_seconds=float(section.get("timeout_seconds", 60)),
    )
    return composer, provider
