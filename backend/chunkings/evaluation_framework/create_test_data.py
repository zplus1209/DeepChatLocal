"""
Convert BEIR-style JSON → questions_df.csv + per-corpus .md files.

KEY FIX: max_span_ratio guard
------------------------------
Answers in this dataset are often **LLM-synthesised** (not verbatim quotes
from the corpus).  The old fuzzy SequenceMatcher found scattered word
fragments across the whole document and built a span 40-80× larger than the
answer, e.g. span=8 036 chars for a 93-char answer.

Downstream, that fake "valid" span made recall near-zero even when the
perfectly correct chunk was retrieved (a 400-char chunk covers only 5% of an
8 000-char span → recall@5 ≈ 0.05).

Fix: after any span-finding attempt, validate:
    (span_end − span_start) / len(answer)  <=  max_span_ratio   (default 3.0)
Spans that fail this check are rejected → start_index = −1.

The evaluation layer (base_evaluation.py) handles start_index = −1 refs with
a text-based fallback so no question is silently ignored.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ═══════════════════════════════════════════════════════════════════════════════
# Whitespace helpers  (must stay in sync with base_evaluation.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_whitespace(ch: str) -> bool:
    if ch in " \t\n\r\xa0":
        return True
    return unicodedata.category(ch) == "Zs"


def _normalise_ws(text: str) -> str:
    return re.sub(r"[\s\xa0]+", " ", text).strip()


def _build_pos_map(raw: str) -> Tuple[str, List[int]]:
    """Collapse whitespace and return (normalised_text, pos_map)."""
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
    joined  = "".join(chars)
    stripped = joined.strip()
    trim     = len(joined) - len(stripped)
    return stripped, positions[trim:] if trim else positions


# ═══════════════════════════════════════════════════════════════════════════════
# Span finder with ratio validation
# ═══════════════════════════════════════════════════════════════════════════════

def _find_span(
    answer: str,
    raw_corpus: str,
    min_block_chars: int = 30,
    min_ratio: float = 0.35,
    max_span_ratio: float = 3.0,
) -> Optional[Tuple[int, int]]:
    """Locate *answer* in *raw_corpus*.

    Returns ``(raw_start, raw_end)`` or ``None``.

    Passes
    ------
    1. Whitespace-normalised exact match.
    2. Fuzzy SequenceMatcher with coverage >= *min_ratio*.

    In both cases we reject the span if::

        (span_end − span_start) / len(answer)  >  max_span_ratio

    This prevents fuzzy matches that scatter across the entire document.
    """
    if not answer.strip() or not raw_corpus.strip():
        return None

    norm_ans    = _normalise_ws(answer)
    norm_corpus, pos_map = _build_pos_map(raw_corpus)

    if not norm_ans or not pos_map:
        return None

    def _ok(s: int, e: int) -> bool:
        return (e - s) / max(len(answer), 1) <= max_span_ratio

    # Pass 1 – exact normalised
    idx = norm_corpus.find(norm_ans)
    if idx != -1:
        rs  = pos_map[idx]
        re_ = pos_map[min(idx + len(norm_ans) - 1, len(pos_map) - 1)] + 1
        return (rs, re_) if _ok(rs, re_) else None

    # Pass 2 – fuzzy
    sm = difflib.SequenceMatcher(None, norm_ans, norm_corpus, autojunk=False)
    good_blocks = [
        (b.a, b.b, b.size)
        for b in sm.get_matching_blocks()
        if b.size >= min_block_chars
    ]
    if not good_blocks:
        return None

    total_matched = sum(b[2] for b in good_blocks)
    if total_matched / max(len(norm_ans), 1) < min_ratio:
        return None

    min_norm_pos = min(b[1]         for b in good_blocks)
    max_norm_pos = max(b[1] + b[2]  for b in good_blocks)
    rs  = pos_map[min_norm_pos]
    re_ = pos_map[min(max_norm_pos - 1, len(pos_map) - 1)] + 1

    return (rs, re_) if _ok(rs, re_) else None


# ═══════════════════════════════════════════════════════════════════════════════
# Main converter
# ═══════════════════════════════════════════════════════════════════════════════

def convert(
    input_path: str,
    output_path: str,
    corpus_md_folder: str,
    min_block_chars: int = 30,
    min_ratio: float = 0.35,
    max_span_ratio: float = 3.0,
    encoding: str = "utf-8",
) -> None:
    with open(input_path, "r", encoding=encoding) as f:
        data = json.load(f)

    queries:       Dict[str, str]       = data["queries"]
    relevant_docs: Dict[str, List[str]] = data["relevant_docs"]
    answers:       Dict[str, str]       = data["answers"]
    corpus:        Dict[str, str]       = data["corpus"]

    # ── Save corpus .md ───────────────────────────────────────────────────────
    md_folder = Path(corpus_md_folder)
    md_folder.mkdir(parents=True, exist_ok=True)
    print("Saving corpus to .md files …")
    for doc_id, content in tqdm(corpus.items()):
        (md_folder / f"{doc_id}.md").write_text(content, encoding="utf-8")
    print(f"Corpus saved: {len(corpus)} files → {md_folder}\n")

    # ── Process QA pairs ─────────────────────────────────────────────────────
    rows: List[Dict] = []
    n_exact    = 0
    n_fuzzy    = 0
    n_rejected = 0
    n_missing  = 0
    n_no_doc   = 0

    for qid, question in tqdm(queries.items(), desc="Processing QA"):
        doc_ids = relevant_docs.get(qid, [])
        answer  = answers.get(qid, "").strip()

        if not doc_ids:
            rows.append({
                "question":   question.strip(),
                "references": json.dumps(
                    [{"content": answer, "start_index": -1, "end_index": -1}],
                    ensure_ascii=False,
                ),
                "corpus_id": "",
            })
            n_no_doc += 1
            continue

        for doc_id in doc_ids:
            raw_corpus = corpus.get(doc_id)
            raw_s = raw_e = -1

            if raw_corpus is not None:
                span = _find_span(
                    answer, raw_corpus,
                    min_block_chars=min_block_chars,
                    min_ratio=min_ratio,
                    max_span_ratio=max_span_ratio,
                )
                if span is not None:
                    raw_s, raw_e = span
                    # Distinguish exact vs fuzzy for reporting
                    norm_ans = _normalise_ws(answer)
                    norm_c, _  = _build_pos_map(raw_corpus)
                    if norm_c.find(norm_ans) != -1:
                        n_exact += 1
                    else:
                        n_fuzzy += 1
                else:
                    # Check whether fuzzy match existed but failed ratio guard
                    norm_ans = _normalise_ws(answer)
                    norm_c, _ = _build_pos_map(raw_corpus)
                    sm = difflib.SequenceMatcher(None, norm_ans, norm_c, autojunk=False)
                    covered = sum(
                        b.size for b in sm.get_matching_blocks()
                        if b.size >= min_block_chars
                    )
                    if covered / max(len(norm_ans), 1) >= min_ratio:
                        n_rejected += 1   # fuzzy match but ratio too large
                    else:
                        n_missing += 1    # genuinely not found

            rows.append({
                "question":   question.strip(),
                "references": json.dumps(
                    [{"content": answer, "start_index": raw_s, "end_index": raw_e}],
                    ensure_ascii=False,
                ),
                "corpus_id": doc_id,
            })

    df = pd.DataFrame(rows, columns=["question", "references", "corpus_id"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8", quoting=1)

    total = len(df)
    valid   = n_exact + n_fuzzy
    invalid = n_rejected + n_missing + n_no_doc
    print("\n✅ Conversion completed")
    print(f"   Total rows               : {total}")
    print(f"   ├─ Exact/norm match      : {n_exact:4d}  ({100*n_exact/max(total,1):.1f}%)")
    print(f"   ├─ Fuzzy match (valid)   : {n_fuzzy:4d}  ({100*n_fuzzy/max(total,1):.1f}%)")
    print(f"   ├─ Fuzzy REJECTED(ratio) : {n_rejected:4d}  ({100*n_rejected/max(total,1):.1f}%)  ← synthesised answer, span too large")
    print(f"   ├─ Not found             : {n_missing:4d}  ({100*n_missing/max(total,1):.1f}%)")
    print(f"   └─ No relevant doc       : {n_no_doc:4d}  ({100*n_no_doc/max(total,1):.1f}%)")
    print(f"\n   Valid span (span-based eval)  : {valid:4d}  ({100*valid/max(total,1):.1f}%)")
    print(f"   Invalid (text-fallback eval)  : {invalid:4d}  ({100*invalid/max(total,1):.1f}%)")
    print(f"\n   CSV → {output_path}")
    print(f"   Corpus .md dir → {corpus_md_folder}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BEIR JSON to questions CSV + corpus .md files"
    )
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--corpus_md", default="./corpora")
    parser.add_argument("--min_ratio",      type=float, default=0.35)
    parser.add_argument("--max_span_ratio", type=float, default=3.0,
                        help="Reject fuzzy spans with size > max_span_ratio × answer_len "
                             "(default 3.0).  Increase to be more permissive, decrease to "
                             "be stricter.")
    parser.add_argument("--min_block", type=int, default=30)
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        corpus_md_folder=args.corpus_md,
        min_block_chars=args.min_block,
        min_ratio=args.min_ratio,
        max_span_ratio=args.max_span_ratio,
    )