"""Hybrid retrieval over a local corpus, with citations and refusal.

Two signals, one ranking:

- Lexical: BM25 over tokens. Catches exact technical terms, which is most of
  what this corpus is made of.
- Semantic: hashed bag-of-words vectors with cosine similarity. No model
  download and no API key, so the pipeline stays runnable offline, and it is
  enough to catch paraphrase (liquidity versus depth of book).

The two are fused with Reciprocal Rank Fusion, which is scale-free: adding a
third scorer does not force rebalancing two arbitrary weights. A reranker pass
then blends in query-term coverage with a small domain synonym table.

Every returned chunk carries its document id, title and a verbatim quote, so a
claim in the final report can always be traced back to the sentence it came
from. If nothing clears the confidence floor the retriever returns an empty
result instead of its best guess, and the report says it has no evidence
rather than inventing one.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .models import Chunk

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]+")

STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the to was were will with
    this these those their there here what which who whom how when where why not no yes
    can could should would may might must shall do does did done than then so such into over
    under about across after before between during each few more most other some only own
    same too very just now also if then
    """.split()
)

# Hand-built and small on purpose. It exists so the hybrid search finds
# paraphrases, and its contents are visible in the eval set so its effect is
# measurable rather than folklore.
ASYMS = (
    ("liquidity", ("depth", "book", "tvl")),
    ("volatility", ("vol", "variance", "swings", "dispersion")),
    ("drawdown", ("decline", "drop", "peak", "trough")),
    ("netflow", ("net-flow", "flow", "inflow", "outflow")),
    ("token", ("coin", "asset", "erc20")),
    ("wallet", ("address", "holder", "account")),
    ("slippage", ("price-impact", "impact", "tolerance")),
    ("marketcap", ("valuation", "market", "cap")),
    ("airdrop", ("distribution", "allocation")),
    ("rug", ("rugpull", "exit-scam", "scam")),
    ("audit", ("review", "verification", "assessment")),
    ("concentration", ("whales", "top-holders")),
    ("dex", ("exchange", "amm", "swap")),
    ("governance", ("dao", "vote", "proposal")),
    ("rewards", ("incentives", "yield", "emissions")),
)
SYNONYMS: dict[str, tuple[str, ...]] = dict(ASYMS)


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def expand(tokens: Iterable[str]) -> list[str]:
    """Add canonical forms for known synonyms so BM25 catches paraphrases."""

    out: list[str] = []
    for token in tokens:
        if token not in STOPWORDS:
            out.append(token)
        for canonical, variants in SYNONYMS.items():
            if token == canonical or token in variants:
                out.append(canonical)
                break
    return out


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""


@dataclass
class Index:
    """A read-only view over chunked documents."""

    chunks: list[Chunk] = field(default_factory=list)
    documents: int = 0

    def __bool__(self) -> bool:
        return bool(self.chunks)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph first, then on size, preferring sentence boundaries.

    A heading is glued to the paragraph under it. A heading that becomes its
    own chunk retrieves as a match for any question containing its words,
    which produces a citation with a title and no content."""

    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size - 1))
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    paragraphs: list[str] = []
    carry = ""
    for block in blocks:
        if block.startswith("#"):
            carry = f"{carry} {block}".strip()
            continue
        paragraphs.append(f"{carry} {block}".strip())
        carry = ""
    if carry:
        paragraphs.append(carry)
    chunks: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            if not current or len(current) + len(sentence) + 1 <= size:
                current = f"{current} {sentence}".strip()
                continue
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()
        if current:
            chunks.append(current)
    if not chunks and text.strip():
        chunks.append(text[:size])
    return chunks


class BM25:
    """Okapi BM25 with the standard k1/b defaults."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                doc_freq[term] += 1
        self.doc_freq = doc_freq
        self.n_docs = len(chunks)

    def idf(self, term: str) -> float:
        n_q = self.doc_freq.get(term, 0)
        if n_q == 0:
            return 0.0
        return math.log(1.0 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))

    def score(self, query_tokens: Sequence[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        if not self.n_docs or not self.avgdl:
            return scores
        for term in query_tokens:
            idf = self.idf(term)
            if idf <= 0:
                continue
            for index, freq in enumerate(self.term_freqs):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avgdl)
                scores[index] += idf * tf * (self.k1 + 1) / denominator
        return scores


def hashed_vector(tokens: Sequence[str], dim: int = 256) -> list[float]:
    """Deterministic hashed bag of words: no model download, no API key."""

    vector = [0.0] * dim
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
        bucket = int(digest[:12], 16) % dim
        sign = 1.0 if int(digest[12:14], 16) % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def rrf_fuse(rankings: Sequence[Sequence[float]], k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion. Ranks only, so scales and weights stay comparable."""

    if not rankings:
        return []
    fused = [0.0] * len(rankings[0])
    for ranking in rankings:
        order = sorted(range(len(ranking)), key=lambda index: ranking[index], reverse=True)
        for rank, index in enumerate(order, start=1):
            fused[index] += 1.0 / (k + rank)
    return fused


def build_index(documents: Sequence[Document],
                chunk_chars: int = 460, chunk_overlap: int = 60, dim: int = 256) -> Index:
    chunks: list[Chunk] = []
    for document in documents:
        for position, text in enumerate(chunk_text(document.text, chunk_chars, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{position}",
                    doc_id=document.doc_id,
                    title=document.title,
                    text=text,
                    score=0.0,
                    lexical_score=0.0,
                    semantic_score=0.0,
                )
            )
    return Index(chunks=chunks, documents=len(documents))


@dataclass
class Retriever:
    """Fuses BM25 with a hashed-vector semantic score, then reranks.

    The refusal threshold is the part that matters. A normalisation of the
    fused score lets the floor be expressed as an interpretable confidence
    fraction rather than a magic constant, and reranker scores are kept on the
    same fused scale so the floor still means something after rescoring.
    """

    index: Index
    dim: int = 256
    rrf_k: int = 60
    top_k: int = 5
    rerank_enabled: bool = True
    refusal_threshold: float = 0.11
    max_evidence: int = 6
    rerank_weight: float = 0.4

    def __post_init__(self) -> None:
        self.bm25 = BM25(self.index.chunks)
        self.vectors = [hashed_vector(tokenize(chunk.text), self.dim) for chunk in self.index.chunks]
        floor_share = self.refusal_threshold if self.refusal_threshold < 1 else self.refusal_threshold / self.top_k
        self._min_fused = floor_share * self.rrf_k

    def search(self, query: str) -> list[Chunk]:
        """Return chunks that clear the floor, best first. Empty means no evidence."""

        query_tokens = expand(tokenize(query))
        if not query_tokens or not self.index.chunks:
            return []
        lexical = self.bm25.score(query_tokens)
        query_vector = hashed_vector(query_tokens, self.dim)
        semantic = [cosine(query_vector, vector) for vector in self.vectors]
        # A chunk is a candidate only if it shares a term with the question.
        # Hash collisions on unrelated text produce small positive cosines
        # (about 0.1 on a three-word query), which is indistinguishable from
        # a real paraphrase match. Lexical overlap is the gate and similarity
        # is the tie-breaker: a question with no term overlap gets no
        # evidence, rather than a number that looks like confidence.
        anchors = [score > 0.0 for score in lexical]
        if not any(anchors):
            return []
        # Scale-fuse: lexical and semantic are comparable in shape, so blending
        # raw scores (rather than ranks) keeps the floor meaningful on small
        # corpora where RRF flattens everything into noise.
        lex_max = max(lexical) or 1.0
        sem_max = max(semantic) or 1.0
        scored: list[Chunk] = []
        for chunk, anchor, lex, sem in zip(self.index.chunks, anchors, lexical, semantic):
            if not anchor:
                continue
            score = 0.5 * (lex / lex_max) + 0.5 * (sem / sem_max)
            if score < 0.5 * self.refusal_threshold:
                continue
            scored.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    score=round(score, 4),
                    lexical_score=round(lex / lex_max, 4),
                    semantic_score=round(sem / sem_max, 4),
                )
            )
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        results = scored[: self.top_k]
        return self.rerank(results, query_tokens)[: self.max_evidence] if self.rerank_enabled else results

    def rerank(self, chunks: Sequence[Chunk], query_tokens: Sequence[str]) -> list[Chunk]:
        """Blend retrieval with query-term coverage of the chunk text.

        Coverage is recomputed after rescoring, and chunks that lose all signal
        under the combined score drop out entirely rather than lingering at the
        bottom with a near-zero confidence.
        """

        if not chunks:
            return []
        unique_terms = set(query_tokens)
        rescored: list[Chunk] = []
        for chunk in chunks:
            chunk_tokens = set(expand(tokenize(chunk.text)))
            coverage = len(unique_terms & chunk_tokens) / max(len(unique_terms), 1)
            blended = (1 - self.rerank_weight) * chunk.score + self.rerank_weight * coverage
            if blended < 0.5 * self.refusal_threshold:
                continue
            rescored.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    score=round(blended, 4),
                    lexical_score=chunk.lexical_score,
                    semantic_score=chunk.semantic_score,
                )
            )
        rescored.sort(key=lambda chunk: chunk.score, reverse=True)
        return rescored


def load_documents(corpus_dir: str | Path) -> list[Document]:
    """Read .md and .txt files as documents. A leading YAML header is stripped."""

    root = Path(corpus_dir)
    documents: list[Document] = []
    if not root.exists():
        return documents
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        if lines and lines[0].strip() == "---":
            lines = lines[1:]
            while lines and lines[0].strip() != "---":
                lines = lines[1:]
            lines = lines[1:] if lines else []
        text = "\n".join(lines).strip()
        if not text:
            continue
        title = path.stem.replace("_", " ").replace("-", " ").title()
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        documents.append(Document(doc_id=path.stem, title=title, text=text, source=str(path)))
    return documents
