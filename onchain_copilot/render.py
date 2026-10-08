"""Report rendering: Markdown, JSON and self-contained HTML.

HTML is written without any CSS or JS dependency so the file opens correctly
offline and survives being emailed to a client. Numbers are laid out in a
table rather than as sentences so a reader can scan and compare, and every
evidence row keeps its doc id and quote so it can be opened and checked.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

from .models import Evidence, Metric, Report, Token


def metric_rows(report: Report) -> list[dict[str, Any]]:
    rows = []
    for metric in report.metrics:
        rows.append({
            "metric": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "note": metric.note,
        })
    return rows


def report_to_dict(report: Report) -> dict[str, Any]:
    """The machine-readable contract. Everything else is a view of this."""

    return {
        "token": {
            "symbol": report.token.symbol,
            "name": report.token.name,
            "chain": report.token.chain,
            "address": report.token.address,
        },
        "generated_at": report.generated_at,
        "question": report.question,
        "headline": report.headline,
        "grounded": report.grounded,
        "flags": list(report.flags),
        "unverified": list(report.unverified),
        "metrics": metric_rows(report),
        "evidence": [
            {
                "claim": item.claim,
                "doc_id": item.doc_id,
                "doc_title": item.doc_title,
                "quote": item.quote,
                "locator": item.locator,
                "score": item.score,
            }
            for item in report.evidence
        ],
        "sections": [
            {"heading": section.heading, "body": section.body}
            for section in report.sections
        ],
        "author": report.author,
    }


def render_json(report: Report) -> str:
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True, default=str)


def render_markdown(report: Report, disclaimer: str = "") -> str:
    lines: list[str] = []
    lines.append(f"# {report.token.symbol} research report")
    lines.append("")
    lines.append(f"Generated {report.generated_at} by {report.author}.")
    lines.append(f"Question: {report.question}")
    lines.append("")
    if report.flags:
        lines.append("Flags: " + ", ".join(report.flags))
        lines.append("")
    for section in report.sections:
        lines.append(f"## {section.heading}")
        lines.append(section.body)
        lines.append("")
    if report.evidence:
        lines.append("## Evidence")
        for index, item in enumerate(report.evidence, start=1):
            lines.append(f"[e{index}] {item.doc_title} (`{item.doc_id}`) - {item.quote}")
        lines.append("")
    if disclaimer:
        lines.append(f"_{disclaimer}_")
        lines.append("")
    return "\n".join(lines)


CSS = """
:root { color-scheme: light dark;
  --bg: #f7f8fa; --card: #ffffff; --ink: #16181d; --muted: #5b6270;
  --line: #d9dee8; --accent: #1f5fbf; --warn: #9a5b00; --quiet: #4c5566; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: 0; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 20px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 12px; font-weight: 600; background: #e7effc; color: var(--accent); }
.badge.ungrounded { background: #fdeede; color: var(--warn); }
section { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: 16px 18px; margin-bottom: 14px; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
  margin: 0 0 10px; color: var(--quiet); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
td.v { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
blockquote { margin: 0; padding: 8px 12px; border-left: 3px solid var(--line);
  color: var(--quiet); font-size: 13px; }
ul { margin: 0; padding-left: 18px; } li { margin-bottom: 6px; }
.foot { color: var(--muted); font-size: 12px; margin-top: 18px; }
"""


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return html.escape(str(value))


def _html_head(report: Report, badge: str) -> str:
    meta = " ".join(html.escape(flag) for flag in report.flags)
    flag_line = f'<p class="sub">Flags: {meta}</p>' if report.flags else ""
    return (
        "<header><h1>" + html.escape(report.token.symbol) + badge + "</h1>"
        + '<p class="sub">' + html.escape(report.headline) + "</p>"
        + '<p class="sub">' + html.escape(report.generated_at)
        + " by " + html.escape(report.author) + "</p>" + flag_line + "</header>"
    )


def _section_html(heading: str, body: str) -> str:
    return f"<section><h2>{html.escape(heading)}</h2>{body}</section>"


def _body_html(body: str) -> str:
    """Markdown-lite: bullet lines, everything else a paragraph."""

    out: list[str] = []
    items: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            items.append("<li>" + html.escape(line[2:].strip()) + "</li>")
        elif line:
            if items:
                out.append("<ul>" + "".join(items) + "</ul>")
                items = []
            out.append("<p style=\"margin:0 0 8px\">" + html.escape(line) + "</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "".join(out)


def write_report(report: Report, out_dir: str | Path, disclaimer: str = "",
                 name: str | None = None) -> dict[str, str]:
    """Write json, md and html. Returns the writing paths by format."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = name or f"{report.token.symbol}-report"
    written: dict[str, str] = {}
    for suffix, payload in (
        ("json", render_json(report)),
        ("md", render_markdown(report, disclaimer)),
        ("html", render_html(report, disclaimer)),
    ):
        path = target / f"{stem}.{suffix}"
        path.write_text(payload, encoding="utf-8")
        written[suffix] = str(path)
    return written
def render_html(report: Report, disclaimer: str = "") -> str:
    """Self-contained HTML. No external CSS, JS or font."""

    badge = "grounded" if report.grounded else "ungrounded"
    badge_html = f'<span class="badge">{badge}</span>'
    parts: list[str] = [_html_head(report, badge_html)]

    for section in report.sections:
        parts.append(_section_html(section.heading, _body_html(section.body)))

    if report.evidence:
        rows = "".join(_evidence_row(index, item) for index, item in enumerate(report.evidence, start=1))
        table = (
            '<table><thead><tr><th style="width:34px">Ref</th>'
            '<th style="width:180px">Source</th><th>Quote</th></tr></thead>'
            "<tbody>" + rows + "</tbody></table>"
        )
        parts.append(_section_html("Evidence", table))

    parts.append(_section_html("Metrics", _metric_table(report.metrics)))

    if disclaimer:
        parts.append(f'<p class="foot">{html.escape(disclaimer)}</p>')

    return (
        HTML_PREFIX
        + "".join(parts)
        + HTML_SUFFIX
    )


def _evidence_row(index: int, item: Evidence) -> str:
    return (
        "<tr><td>e" + str(index) + "</td><td>" + html.escape(item.doc_title)
        + '<div class="sub" style="margin:4px 0 0">' + html.escape(item.doc_id) + "</div></td>"
        "<td>" + html.escape(" ".join(item.quote.split())) + "</td></tr>"
    )


def _metric_table(metrics: Sequence[Metric]) -> str:
    rows = "".join(
        "<tr><td>" + html.escape(metric.name) + '</td><td class="v">'
        + _fmt(metric.value) + "</td><td>" + html.escape(metric.unit) + "</td></tr>"
        for metric in metrics
    )
    return (
        '<table><thead><tr><th>Metric</th><th style="text-align:right">Value</th>'
        "<th>Unit</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


HTML_PREFIX = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    "<style>" + CSS + "</style></head><body><div class=\"wrap\">"
)

HTML_SUFFIX = "</div></body></html>"


