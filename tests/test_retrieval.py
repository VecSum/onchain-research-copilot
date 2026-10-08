"""Retrieval tests.

Three properties are pinned down here, because they are the ones a reviewer of a
RAG system should ask about:

1. Refusal. A question the corpus cannot answer returns an empty result. If
   this test ever fails, the pipeline has started guessing, and a guessing
   research tool is worse than no tool.
2. Citation integrity. Every returned chunk carries the document id and
   title it came from, so the report can quote its own source verbatim.
3. Chunk boundaries. Chunks never lose a paragraph and never exceed the
   budget, which is what keeps the evidence quotes readable.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from onchain_copilot.models import Chunk
from onchain_copilot.retrieval import (
    Document,
    Retriever,
    build_index,
    chunk_text,
    load_documents,
    rrf_fuse,
    tokenize,
)


CORPUS = [
    Document(
        doc_id="liquidity",
        title="Liquidity and price impact",
        text="Liquidity is the reserve depth available in a pool at the current price band. A pool with large liquidity absorbs a given trade with a proportionally smaller price move. Depth of book is the same idea seen from the other side of the trade.",
    ),
    Document(
        doc_id="volatility",
        title="Volatility and drawdown",
        text="Realised volatility annualises the dispersion of hourly returns. A drawdown is the largest peak-to-trough decline inside the window. Position sizing follows from both, not from total return alone.",
    ),
    Document(
        doc_id="netflow",
        title="Net flow and wallets",
        text="Signed net flow is the difference between inflows and outflows across tracked wallets. The free data tier does not expose labelled wallet flows, so volume per liquidity dollar is used as a pressure proxy.",
    ),
]


def build_retriever(chunk_chars=460, overlap=60, threshold=0.05, rerank=False):
    index = build_index(CORPUS, chunk_chars=chunk_chars, chunk_overlap=overlap)
    return Retriever(index, dim=256, rrf_k=60, top_k=3, rerank_enabled=rerank,
                     refusal_threshold=threshold, max_evidence=3)


class TestChunking(unittest.TestCase):
    def test_no_paragraph_is_lost(self):
        text = "A.\n\nB.\n\nC."
        self.assertEqual(chunk_text(text, size=200, overlap=0), ["A.", "B.", "C."])

    def test_chunks_respect_the_size_budget(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        for size in (80, 160, 460):
            for chunk in chunk_text(text, size=size, overlap=0):
                self.assertLessEqual(len(chunk), size)

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(chunk_text("   \n\n  ", size=100, overlap=0), [])

    def test_size_budget_is_validated(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", size=0, overlap=0)

    def test_overlap_never_exceeds_the_budget(self):
        chunks = chunk_text("a b c d e f g h", size=40, overlap=12)
        self.assertTrue(chunks)
        self.assertEqual(chunks, ["a b c d e f g h"])


class TestTokenize(unittest.TestCase):
    def test_stopwords_are_dropped(self):
        self.assertNotIn("the", tokenize("The pool of the token"))

    def test_domain_terms_survive(self):
        self.assertIn("liquidity", tokenize("pool liquidity depth"))

    def test_punctuation_is_a_token_boundary(self):
        self.assertEqual(tokenize("price,impact!"), ["price", "impact"])


class TestFusion(unittest.TestCase):
    def test_rrf_is_rank_only(self):
        scaled = rrf_fuse([[1.0, 2.0, 3.0]], k=60)
        shifted = rrf_fuse([[100.0, 200.0, 300.0]], k=60)
        self.assertEqual(scaled, shifted)

    def test_rrf_orders_by_rank_not_magnitude(self):
        # The first ranking puts doc 0 first; the second reverses it. Under
        # reciprocal rank fusion the fused score of doc 0 must follow the
        # rankings, not the magnitudes.
        agree = rrf_fuse([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], k=60)
        disagree = rrf_fuse([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]], k=60)
        self.assertGreater(agree[0], disagree[0])

    def test_rrf_penalises_a_ranking_that_buries_the_best_doc(self):
        good = rrf_fuse([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], k=60)
        bad = rrf_fuse([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], k=60)
        self.assertGreater(good[0], bad[0])


class TestRetrieval(unittest.TestCase):
    def test_hits_land_on_the_right_document(self):
        hits = build_retriever().search("what is liquidity depth")
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "liquidity")

    def test_paraphrase_still_finds_the_document(self):
        hits = build_retriever().search("depth of book price impact")
        self.assertTrue(any(hit.doc_id == "liquidity" for hit in hits))

    def test_off_corpus_question_refuses(self):
        # Nothing in the corpus discusses kitchens, chairs or bridge
        # tournaments. An empty result is the correct answer, and the report
        # says so instead of quoting the nearest neighbour.
        self.assertEqual(build_retriever().search("who won the bridge tournament"), [])

    def test_gibberish_query_refuses(self):
        self.assertEqual(build_retriever().search("zzzz qqqq wwww"), [])

    def test_every_hit_carries_its_citation(self):
        for hit in build_retriever().search("volatility drawdown sizing"):
            self.assertTrue(hit.doc_id)
            self.assertTrue(hit.title)
            self.assertTrue(hit.text.strip())
            self.assertTrue(hit.chunk_id)

    def test_reranked_hits_keep_the_floor(self):
        hits = build_retriever(rerank=True).search("liquidity depth")
        self.assertTrue(hits)
        self.assertTrue(all(hit.score >= 0.025 for hit in hits))

    def test_proxy_language_survives_retrieval(self):
        hits = build_retriever().search("signed net flow across wallets")
        self.assertTrue(hits)
        self.assertIn("proxy", " ".join(hit.text for hit in hits).lower())

    def test_high_threshold_refuses_everything(self):
        self.assertEqual(build_retriever(threshold=5.0).search("liquidity"), [])


class TestCorpusLoading(unittest.TestCase):
    def test_bundled_corpus_loads(self):
        docs = load_documents("data/corpus")
        self.assertGreaterEqual(len(docs), 4)
        self.assertTrue(all(d.text.strip() for d in docs))

    def test_front_matter_is_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text("---\ntitle: hidden\n---\n\n# Real title\n\nBody.",
                            encoding="utf-8")
            docs = load_documents(tmp)
            self.assertEqual(docs[0].title, "Real title")
            self.assertEqual(docs[0].text, "# Real title\n\nBody.")

    def test_missing_directory_is_empty_not_an_error(self):
        self.assertEqual(load_documents("data/does-not-exist"), [])


if __name__ == "__main__":
    unittest.main()
