---
title: Evaluating a retrieval augmented pipeline
kind: methodology
---

# Evaluating a retrieval augmented pipeline

## A retriever you cannot measure is not a system

Retrieval quality is measured with a small labelled set: a list of questions, each
with the document that should be retrieved and, where relevant, the document that
must not be. Recall at k answers whether the right passage appears at all. Precision
answers whether what appears is relevant. A retriever with high recall and low
precision produces reports that cite the wrong thing confidently, which is worse than
a retriever with low recall, because the reader has no signal that anything is off.

Gold labels are hand written, typically thirty to fifty questions, and they are
versioned alongside the code so a retrieval regression shows up as a failing eval
rather than as a slow loss of trust.

## Faithfulness: did the report stay inside the evidence

A generated report is faithful when every factual claim it makes traces to a cited
passage. The cheap check is coverage: extract the numeric and named entities from the
report text and confirm each one appears in the cited evidence or in the computed
metrics. Any entity outside both sets is a hallucination candidate, and the count is
reported rather than hidden.

## The refusal case is the whole point

The metric that matters most is not average accuracy. It is what the system does when
the corpus contains no evidence for the question. A pipeline that always produces a
fluent paragraph is easy to demo and useless in production, because every absent
answer arrives dressed as a present one. The eval set includes deliberately
unanswerable questions and asserts that the output refuses and cites nothing.

## Latency and where it goes

End to end time is reported per stage: fetch, metric computation, retrieval,
composition and render. The composition stage dominates when a model is involved, and
it is the only stage with real variance, so it is the one worth a cache.

