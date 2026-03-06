"""
BaseChunkingEvaluation – corpus-scoped evaluation for chunking strategies.

Retrieval design
----------------
For each question we retrieve ONLY from the chunks belonging to the same
corpus as the question (using ChromaDB's ``where={"corpus_id": …}`` filter).

This is the correct design for measuring *chunking quality*:
    "Given the correct document, how well do the chunks allow retrieval
    of the answer to a question?"

Mixing all corpora in a flat index measures cross-document retrieval
instead – completely different (and much harder) problem, irrelevant to
comparing chunking strategies.

Reference types
---------------
Type-A  span-based (start_index >= 0)
    Metrics: IoU, span-recall, span-precision, Recall@k, MRR, MAP, nDCG@k.

Type-B  phrase-based (start_index == -1)
    Answer is LLM-synthesised; evaluated by sliding-window phrase matching.
    Metrics: Recall@k (binary), MRR, MAP, nDCG@k.
"""

from __future__ import annotations

import os
import re
import json
import unicodedata
import chromadb
import platform
import numpy as np
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.llms import EmbeddingModel


# ═══════════════════════════════════════════════════════════════════════════════
# Whitespace helpers (keep in sync with create_test_data.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_whitespace(ch: str) -> bool:
    if ch in " \t\n\r\xa0":
        return True
    return unicodedata.category(ch) == "Zs"


def _normalise_ws(text: str) -> str:
    return re.sub(r"[\s\xa0]+", " ", text).strip()


def _build_pos_map(raw: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    positions: List[int] = []
    prev_space = False
    for i, ch in enumerate(raw):
        if _is_whitespace(ch):
            if not prev_space:
                chars.append(" ")
                positions.append(i)
            prev_space = True
        else:
            chars.append(ch)
            positions.append(i)
            prev_space = False
    joined   = "".join(chars)
    stripped = joined.strip()
    trim     = len(joined) - len(stripped)
    return stripped, positions[trim:] if trim else positions


# ═══════════════════════════════════════════════════════════════════════════════
# Phrase extraction for Type-B fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_phrases(text: str, min_chars: int = 18) -> Set[str]:
    """Sliding 4-6 word windows of length >= min_chars."""
    norm  = _normalise_ws(text)
    words = norm.split()
    out: Set[str] = set()
    for w in (4, 5, 6):
        for i in range(len(words) - w + 1):
            p = " ".join(words[i : i + w])
            if len(p) >= min_chars:
                out.add(p)
    return out


def _ref_phrases(references: List[Dict]) -> Set[str]:
    out: Set[str] = set()
    for r in references:
        if r.get("content"):
            out |= _extract_phrases(r["content"])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Range helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sum_of_ranges(ranges: List[Tuple[int, int]]) -> int:
    return sum(max(0, e - s) for s, e in ranges)


def _union_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    sr = sorted(ranges)
    merged = [sr[0]]
    for cs, ce in sr[1:]:
        ls, le = merged[-1]
        if cs <= le:
            merged[-1] = (ls, max(le, ce))
        else:
            merged.append((cs, ce))
    return merged


def _intersect(r1: Tuple[int, int], r2: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    s, e = max(r1[0], r2[0]), min(r1[1], r2[1])
    return (s, e) if s < e else None


def _difference(ranges: List[Tuple[int, int]], tgt: Tuple[int, int]) -> List[Tuple[int, int]]:
    ts, te = tgt
    out = []
    for s, e in ranges:
        if e <= ts or s >= te:
            out.append((s, e))
        elif s < ts and e > te:
            out.extend([(s, ts), (te, e)])
        elif s < ts:
            out.append((s, ts))
        elif e > te:
            out.append((te, e))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk-in-corpus search
# ═══════════════════════════════════════════════════════════════════════════════

def _rigorous_search(corpus: str, chunk: str) -> Tuple[str, int, int]:
    chunk = chunk.strip()
    if not chunk:
        raise ValueError("Empty chunk")
    idx = corpus.find(chunk)
    if idx != -1:
        return chunk, idx, idx + len(chunk)
    nc  = _normalise_ws(chunk)
    nco, pm = _build_pos_map(corpus)
    idx = nco.find(nc)
    if idx != -1:
        s = pm[idx]
        e = pm[min(idx + len(nc) - 1, len(pm) - 1)] + 1
        return chunk, s, e
    raise ValueError(f"Chunk not found in corpus: {chunk[:80]!r}…")


# ═══════════════════════════════════════════════════════════════════════════════
# Reference helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _valid_spans(refs: List[Dict]) -> List[Tuple[int, int]]:
    return [
        (r["start_index"], r["end_index"])
        for r in refs
        if r["start_index"] >= 0 and r["end_index"] > r["start_index"]
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Relevance  (unified Type-A + Type-B)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_relevant(
    meta: Dict,
    chunk_text: str,
    refs: List[Dict],
    phrases: Set[str],
) -> bool:
    """Relevant if span-intersection (A) OR phrase-match (B).

    Note: corpus_id filter is applied BEFORE calling this function
    (all metas already belong to the correct corpus).
    """
    # Type-A
    cs, ce = meta["start_index"], meta["end_index"]
    for rs, re_ in _valid_spans(refs):
        if _intersect((cs, ce), (rs, re_)):
            return True
    # Type-B
    if phrases:
        nc = _normalise_ws(chunk_text)
        if any(p in nc for p in phrases):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Per-question metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _recall_at_k(
    metas: List[Dict], texts: List[str],
    refs: List[Dict], phrases: Set[str], k: int,
) -> float:
    spans = _valid_spans(refs)
    if spans:
        top = metas[:k]
        unused = list(spans)
        for meta in top:
            cs, ce = meta["start_index"], meta["end_index"]
            for rs, re_ in list(unused):
                inter = _intersect((cs, ce), (rs, re_))
                if inter:
                    unused = _difference(unused, inter)
        total   = _sum_of_ranges(spans)
        covered = total - _sum_of_ranges(unused)
        return covered / total if total else 0.0
    else:
        for meta, text in zip(metas[:k], texts[:k]):
            if _is_relevant(meta, text, refs, phrases):
                return 1.0
        return 0.0


def _rr(metas, texts, refs, phrases) -> float:
    for rank, (meta, text) in enumerate(zip(metas, texts), 1):
        if _is_relevant(meta, text, refs, phrases):
            return 1.0 / rank
    return 0.0


def _ap(metas, texts, refs, phrases) -> float:
    n_rel = max(len(_valid_spans(refs)), 1)
    hits, ap = 0, 0.0
    for rank, (meta, text) in enumerate(zip(metas, texts), 1):
        if _is_relevant(meta, text, refs, phrases):
            hits += 1
            ap   += hits / rank
    return ap / n_rel


def _ndcg(metas, texts, refs, phrases, k) -> float:
    def rel(m, t): return float(_is_relevant(m, t, refs, phrases))
    top   = list(zip(metas[:k], texts[:k]))
    dcg   = sum(rel(m, t) / np.log2(i + 2) for i, (m, t) in enumerate(top))
    n_rel = sum(1 for m, t in zip(metas, texts) if rel(m, t) > 0)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / ideal if ideal else 0.0


def _precision_span(metas, refs, k) -> float:
    spans = _valid_spans(refs)
    if not spans:
        return 0.0
    top = metas[:k]
    if not top:
        return 0.0
    ns: List[Tuple[int, int]] = []
    for meta in top:
        cs, ce = meta["start_index"], meta["end_index"]
        for rs, re_ in spans:
            inter = _intersect((cs, ce), (rs, re_))
            if inter:
                ns = _union_ranges([inter] + ns)
    denom = _sum_of_ranges([(m["start_index"], m["end_index"]) for m in top])
    return _sum_of_ranges(ns) / denom if denom else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB adapter
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingModelAdapter(EmbeddingFunction):
    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input))


# ═══════════════════════════════════════════════════════════════════════════════
# BaseChunkingEvaluation
# ═══════════════════════════════════════════════════════════════════════════════

class BaseChunkingEvaluation:
    _RECALL_K    = [1, 3, 5, 10, 20]
    _PRECISION_K = [1, 3, 5, 10, 20]
    _NDCG_K      = [5, 10]

    def __init__(
        self,
        questions_csv_path: str,
        chroma_db_path: Optional[str] = None,
        corpora_id_paths: Optional[Dict[str, str]] = None,
    ) -> None:
        self.questions_csv_path = questions_csv_path
        self.corpora_id_paths   = corpora_id_paths
        self._load_questions_df()
        self._chroma_client = (
            chromadb.PersistentClient(path=chroma_db_path)
            if chroma_db_path else chromadb.Client()
        )

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load_questions_df(self) -> None:
        if not os.path.exists(self.questions_csv_path):
            raise FileNotFoundError(self.questions_csv_path)
        df = pd.read_csv(self.questions_csv_path)
        df["references"] = df["references"].apply(json.loads)
        df["_has_span"]  = df["references"].apply(
            lambda refs: any(
                r["start_index"] >= 0 and r["end_index"] > r["start_index"]
                for r in refs
            )
        )
        df["_phrases"] = df["references"].apply(_ref_phrases)
        self.questions_df = df
        self.corpus_list  = df["corpus_id"].unique().tolist()

    def _read_corpus(self, corpus_id: str) -> str:
        path = corpus_id
        if self.corpora_id_paths:
            path = self.corpora_id_paths.get(corpus_id, corpus_id)
        enc = "utf-8" if platform.system() == "Windows" else None
        with open(path, "r", encoding=enc) as f:
            return f.read()

    # ── Chunking → positions ──────────────────────────────────────────────────

    def _get_chunks_and_metadata(self, chunker) -> Tuple[List[str], List[Dict]]:
        docs, metas = [], []
        for corpus_id in self.corpus_list:
            corpus  = self._read_corpus(corpus_id)
            chunks  = chunker.split_text(corpus)
            skipped = 0
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    _, s, e = _rigorous_search(corpus, chunk)
                except ValueError as exc:
                    skipped += 1
                    print(f"[WARN] {exc}")
                    continue
                docs.append(chunk)
                metas.append({"start_index": s, "end_index": e, "corpus_id": corpus_id})
            if skipped:
                print(f"[WARN] '{corpus_id}': skipped {skipped}/{len(chunks)} chunks.")
        return docs, metas

    # ── Collections ───────────────────────────────────────────────────────────

    def _build_chunk_collection(
        self, chunker, chroma_ef
    ) -> Tuple[chromadb.Collection, List[str]]:
        for name in ("eval_chunks",):
            try:
                self._chroma_client.delete_collection(name)
            except Exception:
                pass
        col = self._chroma_client.create_collection(
            "eval_chunks",
            embedding_function=chroma_ef,
            metadata={"hnsw:search_ef": 50},
        )
        docs, metas = self._get_chunks_and_metadata(chunker)
        if not docs:
            raise RuntimeError("No chunks produced.")
        BATCH = 500
        for i in range(0, len(docs), BATCH):
            bd, bm = docs[i : i + BATCH], metas[i : i + BATCH]
            col.add(documents=bd, metadatas=bm,
                    ids=[str(j) for j in range(i, i + len(bd))])

        # Cache per-corpus chunk lists for omega and corpus-scoped retrieval
        self._corpus_chunks: Dict[str, Tuple[List[Dict], List[str]]] = {}
        for meta, doc in zip(metas, docs):
            cid = meta["corpus_id"]
            if cid not in self._corpus_chunks:
                self._corpus_chunks[cid] = ([], [])
            self._corpus_chunks[cid][0].append(meta)
            self._corpus_chunks[cid][1].append(doc)

        print(f"[INFO] {col.count()} chunks indexed across {len(self._corpus_chunks)} corpora.")
        return col, docs

    def _build_question_collection(self, chroma_ef) -> chromadb.Collection:
        try:
            self._chroma_client.delete_collection("eval_questions")
        except Exception:
            pass
        col = self._chroma_client.create_collection(
            "eval_questions",
            embedding_function=chroma_ef,
            metadata={"hnsw:search_ef": 50},
        )
        col.add(
            documents=self.questions_df["question"].tolist(),
            metadatas=[{"corpus_id": c} for c in self.questions_df["corpus_id"]],
            ids=[str(i) for i in self.questions_df.index],
        )
        return col

    # ── Precision-omega (ceiling) ─────────────────────────────────────────────

    def _precision_omega(self) -> Tuple[List[float], List[int]]:
        scores, hcc = [], []
        for _, row in self.questions_df.iterrows():
            refs      = row["references"]
            corpus_id = row["corpus_id"]
            phrases   = row["_phrases"]
            spans     = _valid_spans(refs)

            c_metas, c_texts = self._corpus_chunks.get(corpus_id, ([], []))

            if spans:
                unused = list(spans)
                num_sets: List[Tuple[int, int]] = []
                den_sets: List[Tuple[int, int]] = []
                count = 0
                for meta in c_metas:
                    cs, ce = meta["start_index"], meta["end_index"]
                    hit = False
                    for rs, re_ in spans:
                        inter = _intersect((cs, ce), (rs, re_))
                        if inter:
                            hit      = True
                            unused   = _difference(unused, inter)
                            num_sets = _union_ranges([inter]    + num_sets)
                            den_sets = _union_ranges([(cs, ce)] + den_sets)
                    if hit:
                        count += 1
                den_sets = _union_ranges(den_sets + unused)
                score = (
                    _sum_of_ranges(num_sets) / _sum_of_ranges(den_sets)
                    if num_sets else 0.0
                )
                scores.append(score)
                hcc.append(count)
            else:
                hit_count = sum(
                    1 for meta, text in zip(c_metas, c_texts)
                    if any(p in _normalise_ws(text) for p in phrases)
                )
                scores.append(1.0 if hit_count > 0 else 0.0)
                hcc.append(hit_count)

        return scores, hcc

    # ── Corpus-scoped retrieval ────────────────────────────────────────────────

    def _retrieve_for_corpus(
        self,
        chunk_col: chromadb.Collection,
        question_embs: Dict[int, List[float]],
        corpus_id: str,
        question_indices: List[int],
        n_results: int,
    ) -> Dict[int, Tuple[List[Dict], List[str]]]:
        """Retrieve top-n chunks filtered to corpus_id for a batch of questions.

        Returns {question_idx: (metas, texts)}.
        """
        n_in_corpus = len(self._corpus_chunks.get(corpus_id, ([], []))[0])
        if n_in_corpus == 0:
            return {i: ([], []) for i in question_indices}

        k = min(n_results, n_in_corpus)
        embs = [question_embs[i] for i in question_indices]

        results = chunk_col.query(
            query_embeddings=embs,
            n_results=k,
            where={"corpus_id": corpus_id},
            include=["metadatas", "documents"],
        )

        out: Dict[int, Tuple[List[Dict], List[str]]] = {}
        for local_i, global_i in enumerate(question_indices):
            out[global_i] = (
                results["metadatas"][local_i],
                results["documents"][local_i],
            )
        return out

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        chunker,
        embedding_model: EmbeddingModel,
        retrieve: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate chunker with corpus-scoped retrieval.

        For each question, only chunks from the correct corpus are candidates.
        This measures chunking quality, not cross-document retrieval.
        """
        self._load_questions_df()
        self.questions_df = self.questions_df.sort_index()

        chroma_ef  = EmbeddingModelAdapter(embedding_model)
        chunk_col, all_chunk_texts = self._build_chunk_collection(chunker, chroma_ef)
        question_col               = self._build_question_collection(chroma_ef)

        omega_scores, highlighted  = self._precision_omega()

        # ── max_k ─────────────────────────────────────────────────────────────
        if retrieve == -1:
            effective_k = [max(1, hc) for hc in highlighted]
        else:
            effective_k = [retrieve] * len(highlighted)

        max_k = max(
            max(self._RECALL_K + self._PRECISION_K + self._NDCG_K),
            max(effective_k),
        )

        # ── Embed all questions ────────────────────────────────────────────────
        q_data = question_col.get(include=["embeddings"])
        q_embs: Dict[int, List[float]] = {
            int(qid): emb
            for qid, emb in zip(q_data["ids"], q_data["embeddings"])
        }

        # ── Group questions by corpus for batch retrieval ──────────────────────
        # corpus_id → list of (original_df_position, df_index)
        from collections import defaultdict
        corpus_to_rows: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for pos, (df_idx, row) in enumerate(self.questions_df.iterrows()):
            corpus_to_rows[row["corpus_id"]].append((pos, df_idx))

        # ── Corpus-scoped retrieval for all questions ──────────────────────────
        # result_by_pos[pos] = (metas, texts)
        result_by_pos: Dict[int, Tuple[List[Dict], List[str]]] = {}

        for corpus_id, pos_idx_pairs in corpus_to_rows.items():
            positions  = [p for p, _ in pos_idx_pairs]
            df_indices = [i for _, i in pos_idx_pairs]

            per_q = self._retrieve_for_corpus(
                chunk_col, q_embs,
                corpus_id, df_indices,
                n_results=max_k,
            )
            for pos, df_idx in zip(positions, df_indices):
                result_by_pos[pos] = per_q.get(df_idx, ([], []))

        # ── Accumulate metrics ────────────────────────────────────────────────
        iou_sc  = [];  valid_iou = []
        rec_sc  = [];  valid_rec = []
        pre_sc  = [];  valid_pre = []
        rr_sc   = []
        ap_sc   = []
        recall_at:    Dict[int, List[float]] = {k: [] for k in self._RECALL_K}
        precision_at: Dict[int, List[float]] = {k: [] for k in self._PRECISION_K}
        ndcg_at:      Dict[int, List[float]] = {k: [] for k in self._NDCG_K}
        corpora_scores: Dict[str, Dict] = {}

        for pos, (_, row) in enumerate(self.questions_df.iterrows()):
            refs      = row["references"]
            corpus_id = row["corpus_id"]
            phrases   = row["_phrases"]
            metas, texts = result_by_pos.get(pos, ([], []))
            k         = effective_k[pos]
            spans     = _valid_spans(refs)

            # ── Span-based (Type-A) ───────────────────────────────────────────
            if spans:
                unused = list(spans)
                num_sets: List[Tuple[int, int]] = []
                den_sets: List[Tuple[int, int]] = []

                for meta in metas[:k]:
                    cs, ce = meta["start_index"], meta["end_index"]
                    for rs, re_ in spans:
                        inter = _intersect((cs, ce), (rs, re_))
                        if inter:
                            unused   = _difference(unused, inter)
                            num_sets = _union_ranges([inter]    + num_sets)
                            den_sets = _union_ranges([(cs, ce)] + den_sets)

                num_val  = _sum_of_ranges(num_sets)
                ref_span = _sum_of_ranges(spans)
                ret_span = _sum_of_ranges([(m["start_index"], m["end_index"]) for m in metas[:k]])
                iou_den  = ret_span + _sum_of_ranges(unused)

                rec  = num_val / ref_span if ref_span else 0.0
                prec = num_val / ret_span if ret_span else 0.0
                iou  = num_val / iou_den  if iou_den  else 0.0
                iou_sc.append(iou);  valid_iou.append(iou)
                rec_sc.append(rec);  valid_rec.append(rec)
                pre_sc.append(prec); valid_pre.append(prec)
            else:
                iou_sc.append(0.0)
                rec_sc.append(0.0)
                pre_sc.append(0.0)

            # ── Ranking (both types) ─────────────────────────────────────────
            for kv in self._RECALL_K:
                recall_at[kv].append(_recall_at_k(metas, texts, refs, phrases, kv))
            for kv in self._NDCG_K:
                ndcg_at[kv].append(_ndcg(metas, texts, refs, phrases, kv))
            for kv in self._PRECISION_K:
                precision_at[kv].append(_precision_span(metas, refs, kv))
            rr_sc.append(_rr(metas, texts, refs, phrases))
            ap_sc.append(_ap(metas, texts, refs, phrases))

            # ── Per-corpus ───────────────────────────────────────────────────
            if corpus_id not in corpora_scores:
                corpora_scores[corpus_id] = {
                    "iou_scores": [], "recall_scores": [], "precision_scores": [],
                    "precision_omega_scores": [], "mrr": [], "map": [],
                    **{f"recall@{k}":    [] for k in self._RECALL_K},
                    **{f"precision@{k}": [] for k in self._PRECISION_K},
                    **{f"ndcg@{k}":      [] for k in self._NDCG_K},
                }
            cd = corpora_scores[corpus_id]
            cd["iou_scores"].append(iou_sc[-1])
            cd["recall_scores"].append(rec_sc[-1])
            cd["precision_scores"].append(pre_sc[-1])
            cd["precision_omega_scores"].append(omega_scores[pos])
            cd["mrr"].append(rr_sc[-1])
            cd["map"].append(ap_sc[-1])
            for kv in self._RECALL_K:
                cd[f"recall@{kv}"].append(recall_at[kv][-1])
            for kv in self._PRECISION_K:
                cd[f"precision@{kv}"].append(precision_at[kv][-1])
            for kv in self._NDCG_K:
                cd[f"ndcg@{kv}"].append(ndcg_at[kv][-1])

        def _mean(lst): return float(np.mean(lst)) if lst else 0.0
        def _std(lst):  return float(np.std(lst))  if lst else 0.0

        n_total = len(self.questions_df)
        n_span  = int(self.questions_df["_has_span"].sum())

        return {
            "n_questions":      n_total,
            "n_span_refs":      n_span,
            "n_text_refs":      n_total - n_span,
            "n_chunks_indexed": chunk_col.count(),
            "n_corpora":        len(self._corpus_chunks),

            # Span-based (all rows; 0 for Type-B)
            "iou_mean":             _mean(iou_sc),
            "iou_std":              _std(iou_sc),
            "recall_mean":          _mean(rec_sc),
            "recall_std":           _std(rec_sc),
            "precision_mean":       _mean(pre_sc),
            "precision_std":        _std(pre_sc),
            "precision_omega_mean": _mean(omega_scores),
            "precision_omega_std":  _std(omega_scores),

            # Span-based (Type-A rows only)
            "iou_mean_spanonly":       _mean(valid_iou),
            "recall_mean_spanonly":    _mean(valid_rec),
            "precision_mean_spanonly": _mean(valid_pre),

            # Ranking (all rows, Type-A + Type-B)
            "mrr": _mean(rr_sc),
            "map": _mean(ap_sc),
            **{f"recall@{k}":    _mean(recall_at[k])    for k in self._RECALL_K},
            **{f"precision@{k}": _mean(precision_at[k]) for k in self._PRECISION_K},
            **{f"ndcg@{k}":      _mean(ndcg_at[k])      for k in self._NDCG_K},

            "corpora_scores": corpora_scores,
        }