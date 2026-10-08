# onchain-research-copilot

Config-driven on-chain research copilot. Point it at a pool address and a
question, and it produces a structured research report: reproducible
quantitative metrics computed from OHLCV and snapshot data, cited evidence
retrieved from a research corpus, and a refusal when the corpus has nothing to
say. Runs fully offline with zero third-party dependencies.

    pool address + question
            |
            v
    [providers]  adapter per source, local snapshot cache, offline fixtures
            |
            v
    [metrics]    18 quantitative measures, each pure and unit-tested
            |
            v
    [retrieval]  hybrid BM25 + dense, RRF fusion, reranker, refusal threshold
            |
            v
    [composer]   deterministic template or any OpenAI-compatible endpoint
            |
            v
    report.{json,md,html}  measured behaviour, cited evidence, open questions

## Why it is shaped this way

Most AI trading repos are a strategy class wrapped around an LLM call. This one
takes the opposite position: the numbers are computed by deterministic code that
is unit-tested against reference implementations, and the LLM only narrates them.
That separation is what makes the report auditable, and it is why the evaluation
harness can assert that every figure in the prose traces back to a metric the
pipeline actually computed.

Three properties are treated as features rather than fallbacks:

**Degradation is a result.** When a stage cannot produce its output the run
continues with a recorded reason. A missing evidence set produces an explicit
refusal, a failed composer falls back to the deterministic one, and both are
visible in the report instead of a silent blank.

**Refusal beats confidence.** The retriever has a score floor. A question with
no supporting document returns no evidence, and the report says so. A confident
citation of the nearest neighbour is the failure mode this guards against.

**Offline by default.** data_mode = fixtures and provider = deterministic mean a
fresh clone produces a real report with no network and no API key.

## Quick start

Requires Python 3.11+ (uses tomllib). No dependencies to install.

    git clone https://github.com/VecSum/onchain-research-copilot.git
    cd onchain-research-copilot

    make test    # 41 unit tests, metrics and retrieval
    make eval    # end-to-end evaluation report -> output/eval-report.md
    make demo    # one offline report -> output/DEMO-report.{json,md,html}

To generate a report for a specific pool and question:

python3 -m onchain_copilot research \
      --pool 0xdemo000000000000000000000000000000000001 \
      --question "how much liquidity does the pool have" \
      --offline --print-report

Writes JSON, Markdown and a self-contained HTML into output/, named after the
pool symbol (output/DEMO-report.json for the bundled fixture pool).

--offline uses the bundled fixtures. Drop it and the configured providers are
used instead (see Configuration below).

## Evaluation

make eval scores every case in eval/dataset.jsonl individually, so a regression
names the case that broke instead of hiding inside an average.

| Metric | Value | What it guards against |
| --- | --- | --- |
| Retrieval Recall@3 | 100% | the document that answers the question never surfaces |
| Refusal accuracy | 100% | a confident citation of the nearest neighbour |
| Grounded rate | 67% | the report asserts a claim with no evidence behind it |
| Faithfulness | 100% | a figure in the prose that no metric computed |

Grounded rate sits below 100% by design, not by failure: the dataset
deliberately contains off-corpus questions, and the correct behaviour for those
is to refuse. Faithfulness re-parses every numeric token in the report prose and
requires it to match a computed metric value within rounding. Numbers inside a
verbatim evidence quote are exempt, because those words came from the corpus and
the composer cannot edit them.

## Metrics

All computed in onchain_copilot/metrics.py, each a pure function over a candle
sequence or a pool snapshot, each covered by a test that compares against an
independently written reference implementation.

Price and return: close, window high and low, window change, annualised realised
volatility from hourly returns, return z-score, RSI, max drawdown, fast and slow
momentum, upside and downside capture, risk/reward asymmetry.

Flow and structure: volume ratio against a trailing mean, volume per liquidity
dollar as a pressure proxy, FDV to liquidity, liquidity and 24h volume.

Two of these deserve a note. Upside and downside capture is undefined unless the
window contains both up and down moves, so the pipeline records
capture_undefined rather than emitting a misleading zero. And volume per
liquidity is labelled a pressure proxy everywhere it appears, because a free
data tier reports volume but not signed net flow, so direction is genuinely
unknown. Both gaps are stated in the report instead of being smoothed over.

## Retrieval

Hybrid retrieval over the local corpus: BM25 lexical scoring and a dense cosine
channel fused with reciprocal rank fusion, then an optional reranking pass.
Chunking is character-based with overlap, and every chunk keeps its document id
and a locator so a citation can be opened and checked.

The evidence bound to a claim is the sentence nearest the highest-scoring match,
not a paraphrase of it. A reader checking a citation sees the source's own words.

## Configuration

Copy config.example.toml to config.toml. Every key has a working default, so the
file is optional. The four sections that matter:

| Section | Purpose |
| --- | --- |
| [run] | synthetic, fixtures or live data mode |
| [providers] | provider order, cache dir, TTL, timeout, retries |
| [retrieval] | corpus dir, chunking, fusion weights, refusal threshold |
| [llm] | deterministic or openai_compat, with fallback enabled |

provider = openai_compat points at any OpenAI-compatible Chat Completions
endpoint. If the call fails or the key is missing, the run falls back to the
deterministic composer and records composer_degraded rather than failing.

## Layout

    onchain_copilot/
      providers.py   one adapter per data source, cache, retry, health probe
      metrics.py     the quantitative layer, pure functions only
      retrieval.py   chunking, hybrid index, RRF fusion, reranker, refusal
      llm.py         deterministic composer and OpenAI-compatible composer
      pipeline.py    the linear orchestration and degradation recording
      render.py      JSON, Markdown and self-contained HTML
      config.py      typed configuration over tomllib
    tests/           unit tests for metrics and retrieval
    eval/            the evaluation harness and its dataset
    data/corpus/     research documents the retriever reads
    data/fixtures/   offline pool snapshots

## Scope

This is a research and reporting tool. It does not place orders, hold keys,
connect to an exchange account, or produce buy or sell recommendations. Every
report carries the disclaimer that it is a research artefact and not a
recommendation to transact. The metrics layer is deliberately read-only against
price and liquidity data.

Second-level streaming is out of scope for the same reason: a free data tier
cannot support it, so snapshots are minute-scale and the timestamp precision is
stated honestly rather than implied.

## License

MIT. See LICENSE.
