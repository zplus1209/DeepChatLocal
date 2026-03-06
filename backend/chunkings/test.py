"""
Full chunking experiment – RecursiveTokenChunker, ClusterSemanticChunker, LLMSemanticChunker.

Run:
    CUDA_VISIBLE_DEVICES=1 python backend/chunkings/test.py
"""

import json
import logging
import sys
import time
import textwrap
import random
from datetime import datetime
from pathlib import Path
from itertools import product

from backend.chunkings.evaluation_framework.general_evaluation import (
    GeneralChunkingEvaluation,
)
from backend.chunkings import (
    RecursiveTokenChunker,
    ClusterSemanticChunker,
    LLMSemanticChunker,
)
from backend.llms import EmbeddingModel


RECURSIVE_SIZES    = [200, 400, 800, 1000, 1200]
RECURSIVE_OVERLAPS = [0, 50, 100, 150, 200, 250]
CLUSTER_MAX_SIZES  = [200, 400, 800, 1000, 1200]

RESULT_DIR = Path("./full_chunk_experiment")
VISUAL_DIR = RESULT_DIR / "chunk_visual"
RESULT_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)

PRIMARY_METRIC = "recall@5"
RUN_TS         = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH       = RESULT_DIR / f"run_{RUN_TS}.log"


_fmt = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_fmt.default_msec_format = "%s.%03d"

log = logging.getLogger("chunk_exp")
log.setLevel(logging.DEBUG)

_fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
log.addHandler(_fh)

_sh = logging.StreamHandler(sys.stderr)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
log.addHandler(_sh)


evaluation      = GeneralChunkingEvaluation(chroma_db_path="./chroma_db")
embedding_model = EmbeddingModel(
    engine="hf",
    model_name="dangvantuan/vietnamese-document-embedding",
)
all_results: list = []
experiment_start  = time.time()

log.info("=" * 70)
log.info(f"EXPERIMENT  {RUN_TS}")
log.info(f"log  -> {LOG_PATH}")
log.info(f"dir  -> {RESULT_DIR.resolve()}")
log.info(f"recursive  : sizes={RECURSIVE_SIZES}  overlaps={RECURSIVE_OVERLAPS}")
log.info(f"cluster    : max_sizes={CLUSTER_MAX_SIZES}")
log.info("=" * 70)


def run_and_save(chunker, config_name: str, config_dict: dict) -> None:
    log.info("-" * 60)
    log.info(f"START   {config_name}")
    log.info(f"config  {json.dumps(config_dict, ensure_ascii=False)}")

    t0       = time.time()
    ts_start = datetime.now().isoformat(timespec="milliseconds")
    results  = evaluation.run(chunker, embedding_model)
    t1       = time.time()
    elapsed  = round(t1 - t0, 3)
    ts_end   = datetime.now().isoformat(timespec="milliseconds")

    results["config"]             = config_dict
    results["runtime_seconds"]    = elapsed
    results["timestamp_start"]    = ts_start
    results["timestamp_end"]      = ts_end
    results["wall_clock_start_s"] = round(t0 - experiment_start, 3)
    results["wall_clock_end_s"]   = round(t1 - experiment_start, 3)

    out_path = RESULT_DIR / f"{config_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    all_results.append(results)

    log.info(f"  done    [{ts_start} -> {ts_end}]  {elapsed}s")
    log.info(f"  chunks  : {results.get('n_chunks_indexed','?')}")
    log.info(f"  refs    : {results.get('n_valid_ref','?')} / {results.get('n_questions','?')}")
    log.info(f"  recall@5        : {results.get('recall@5',  0):.4f}")
    log.info(f"  recall@10       : {results.get('recall@10', 0):.4f}")
    log.info(f"  MRR             : {results.get('mrr',       0):.4f}")
    log.info(f"  nDCG@10         : {results.get('ndcg@10',   0):.4f}")
    log.info(f"  precision_omega : {results.get('precision_omega_mean', 0):.4f}")
    log.info(f"  recall_valid    : {results.get('recall_mean_valid',    0):.4f}")


def load_corpus_texts() -> dict[str, str]:
    # corpora_id_paths: {doc_id: filepath} – được set trong __init__ của GeneralChunkingEvaluation
    corpus_texts = {}
    for doc_id, filepath in evaluation.corpora_id_paths.items():
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        corpus_texts[doc_id] = text
    return corpus_texts


def run_visual_comparison() -> Path:
    vis_log_path = VISUAL_DIR / f"visual_{RUN_TS}.log"

    vis_log = logging.getLogger("chunk_visual")
    vis_log.setLevel(logging.DEBUG)
    _vfh = logging.FileHandler(vis_log_path, encoding="utf-8")
    _vfh.setFormatter(logging.Formatter("%(message)s"))
    vis_log.addHandler(_vfh)

    visual_chunkers = {
        "Recursive(200, overlap=0)  " : RecursiveTokenChunker(chunk_size=200, chunk_overlap=0),
        "Recursive(400, overlap=100)" : RecursiveTokenChunker(chunk_size=400, chunk_overlap=100),
        "ClusterSemantic(max=200)   " : ClusterSemanticChunker(max_chunk_size=200),
    }

    corpus_texts = load_corpus_texts()
    random.seed(42)
    samples = random.sample(list(corpus_texts.items()), min(10, len(corpus_texts)))

    log.info(f"visual: {len(corpus_texts)} docs in corpus, sampling {len(samples)}")

    vis_log.info("=" * 90)
    vis_log.info(f"VISUAL CHUNKING COMPARISON  {RUN_TS}")
    vis_log.info("chunkers: " + " | ".join(k.strip() for k in visual_chunkers))
    vis_log.info("=" * 90)

    for doc_id, text in samples:
        vis_log.info("")
        vis_log.info("#" * 90)
        vis_log.info(f"  DOC [{doc_id}]   {len(text)} chars / ~{len(text.split())} words")
        vis_log.info("#" * 90)
        vis_log.info("")
        vis_log.info("-- ORIGINAL " + "-" * 76)
        for line in textwrap.wrap(text[:800], width=82):
            vis_log.info(f"   {line}")
        if len(text) > 800:
            vis_log.info("   [... truncated for display ...]")
        vis_log.info("")

        stats = {}
        for label, chunker in visual_chunkers.items():
            t0     = time.time()
            chunks = chunker.split_text(text)
            dur_ms = round((time.time() - t0) * 1000, 1)
            avg_w  = round(sum(len(c.split()) for c in chunks) / max(len(chunks), 1), 1)
            stats[label] = {"n": len(chunks), "avg": avg_w, "ms": dur_ms}

            vis_log.info(f"-- {label.strip()}  [{len(chunks)} chunks  {dur_ms}ms] " + "-" * 20)
            for i, chunk in enumerate(chunks):
                n_w     = len(chunk.split())
                preview = chunk.strip()[:200]
                suffix  = " ..." if len(chunk.strip()) > 200 else ""
                vis_log.info(f"  [chunk {i+1:02d} ~{n_w}w] {preview}{suffix}")
            vis_log.info("")

        vis_log.info("-- SUMMARY " + "-" * 77)
        vis_log.info(f"  {'chunker':<30} | {'n_chunks':>8} | {'avg_words':>9} | {'ms':>8}")
        vis_log.info(f"  {'-'*60}")
        for label, st in stats.items():
            vis_log.info(f"  {label.strip():<30} | {st['n']:>8} | {st['avg']:>9} | {st['ms']:>8}")
        vis_log.info("")

        log.debug(f"  visual [{doc_id}] done")

    vis_log.info("=" * 90)
    log.info(f"visual -> {vis_log_path}")
    return vis_log_path


vis_path = run_visual_comparison()


log.info("=" * 60)
log.info("RecursiveTokenChunker")
log.info("=" * 60)

for size, overlap in product(RECURSIVE_SIZES, RECURSIVE_OVERLAPS):
    if overlap >= size:
        log.info(f"skip  size={size} overlap={overlap}")
        continue
    chunker = RecursiveTokenChunker(chunk_size=size, chunk_overlap=overlap)
    run_and_save(
        chunker,
        config_name=f"recursive_{size}_{overlap}",
        config_dict={"chunker": "RecursiveTokenChunker", "chunk_size": size, "chunk_overlap": overlap},
    )


log.info("=" * 60)
log.info("ClusterSemanticChunker")
log.info("=" * 60)

for max_size in CLUSTER_MAX_SIZES:
    chunker = ClusterSemanticChunker(max_chunk_size=max_size)
    run_and_save(
        chunker,
        config_name=f"cluster_{max_size}",
        config_dict={"chunker": "ClusterSemanticChunker", "max_chunk_size": max_size},
    )


log.info("=" * 60)
log.info("LLMSemanticChunker")
log.info("=" * 60)

chunker = LLMSemanticChunker()
run_and_save(
    chunker,
    config_name="llm_semantic",
    config_dict={"chunker": "LLMSemanticChunker"},
)


summary_path = RESULT_DIR / "summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

all_results.sort(key=lambda r: r.get(PRIMARY_METRIC, 0), reverse=True)
total_elapsed = round(time.time() - experiment_start, 2)

log.info("")
log.info("=" * 100)
log.info("SUMMARY TABLE  sorted by recall@5 desc")
log.info("=" * 100)

hdr = (
    f"{'Config':<30} | {'chunks':>7} | "
    f"{'R@5':>6} | {'R@10':>6} | {'MRR':>6} | "
    f"{'nDCG@10':>8} | {'prec_w':>7} | "
    f"{'recall_v':>9} | {'start_s':>8} | {'dur_s':>7}"
)
log.info(hdr)
log.info("-" * len(hdr))

for r in all_results:
    cfg  = r["config"]
    name = cfg["chunker"]
    if "chunk_size"     in cfg: name += f"/{cfg['chunk_size']}"
    if "chunk_overlap"  in cfg: name += f"/{cfg['chunk_overlap']}"
    if "max_chunk_size" in cfg: name += f"/{cfg['max_chunk_size']}"

    log.info(
        f"{name:<30} | "
        f"{r.get('n_chunks_indexed', 0):>7} | "
        f"{r.get('recall@5',   0):.4f} | "
        f"{r.get('recall@10',  0):.4f} | "
        f"{r.get('mrr',        0):.4f} | "
        f"{r.get('ndcg@10',    0):.4f}   | "
        f"{r.get('precision_omega_mean', 0):.4f}  | "
        f"{r.get('recall_mean_valid',    0):.4f}    | "
        f"{r.get('wall_clock_start_s',   0):>8.1f} | "
        f"{r.get('runtime_seconds',      0):>7.1f}"
    )

log.info("")
log.info(f"total time : {total_elapsed}s  ({total_elapsed/60:.1f} min)")
log.info(f"configs    : {len(all_results)}")
log.info(f"log        : {LOG_PATH}")
log.info(f"visual     : {vis_path}")
log.info(f"summary    : {summary_path}")