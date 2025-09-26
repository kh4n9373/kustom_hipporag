#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare HippoRAG DPR retrieval outputs using SeCom segmentation as preprocessing.

Flow:
  - For every conversation in the dataset, take its `chunks` (list of strings)
  - For each chunk, split into sentence-like "exchanges" and call SeCom.segment
    to segment into topical subchunks (lists of exchanges)
  - Build mapping mother_chunk -> [subchunks...] and subchunk -> mother_chunk
  - Index all subchunks into HippoRAG (passage store only)
  - For every question, run DPR retrieval; map retrieved subchunks back to original chunks
  - Save output JSON compatible with evaluator.py (retrieved list contains original chunks)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# Ensure project-local import works when running from repo root
from src.hipporag import HippoRAG  # type: ignore
from src.hipporag.utils.config_utils import BaseConfig  # type: ignore


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:1]


def sentence_split(text: str) -> List[str]:
    # Lightweight sentence split to form SeCom exchanges
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    # Split on punctuation followed by whitespace/newline; keep simple to avoid extra deps
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    # Filter out very short fragments; keep punctuation
    exchanges = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return exchanges if exchanges else [text]


def build_secom_segments(
    chunks: List[str],
    secom_config_path: str | None = None,
    debug_path: str | None = None,
    use_window: bool = False,
    window_size: int = 5,
    window_overlap: int = 2,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      mother_to_subs: original chunk -> list of subchunks (strings)
      sub_to_mother: subchunk -> original chunk
    """
    # Import SeCom from source tree without requiring installation
    secom_repo_dir = os.path.join(os.getcwd(), "SeCom")
    if os.path.isdir(secom_repo_dir) and secom_repo_dir not in sys.path:
        sys.path.insert(0, secom_repo_dir)
    try:
        from secom.secom import SeCom  # type: ignore
    except Exception:
        # Fallback import path used in tests (from SeCom.secom import SeCom)
        from SeCom.secom import SeCom  # type: ignore

    memory_manager = SeCom(
        granularity="segment",
        config_path=(
            secom_config_path
            if secom_config_path
            else os.path.join(secom_repo_dir, "secom", "configs", "mpnet.yaml")
        ),
    )

    mother_to_subs: Dict[str, List[str]] = {}
    sub_to_mother: Dict[str, str] = {}

    # Optional debug log file (JSONL)
    dbg_f = open(debug_path, "a", encoding="utf-8") if debug_path else None

    def _window_segments(exchanges: List[str]) -> List[str]:
        if window_size <= 1:
            return ["\n".join([ex]) for ex in exchanges]
        step = max(1, window_size - max(0, window_overlap))
        out: List[str] = []
        for start in range(0, len(exchanges), step):
            seg = exchanges[start:start + window_size]
            if not seg:
                break
            out.append("\n".join(seg))
            if start + window_size >= len(exchanges):
                break
        return out

    for idx, chunk in enumerate(chunks):
        exchanges = sentence_split(chunk)
        # SeCom.segment expects a list of sessions; one session is list of exchanges
        segments = memory_manager.segment([exchanges])
        # For one session, SeCom returns a list of segment lists (each list is a segment of exchanges)
        # We will join each segment into a subchunk string
        subchunks: List[str] = [
            "\n".join(seg) if isinstance(seg, list) else str(seg) for seg in segments
        ]
        # Optional sliding-window augmentation
        if use_window:
            subchunks.extend(_window_segments(exchanges))
        # If segmentation unexpectedly returned an empty list, fallback to the whole chunk
        if not subchunks:
            subchunks = [chunk]

        mother_to_subs[chunk] = subchunks
        for sub in subchunks:
            # If duplicates appear, keep first mapping
            sub_to_mother.setdefault(sub, chunk)

        if dbg_f:
            json.dump({
                "type": "segmentation",
                "chunk_idx": idx,
                "num_exchanges": len(exchanges),
                "num_subchunks": len(subchunks)
            }, dbg_f, ensure_ascii=False)
            dbg_f.write("\n")

    if dbg_f:
        dbg_f.close()

    return mother_to_subs, sub_to_mother


def collect_original_chunks(dataset: List[Dict[str, Any]]) -> List[str]:
    corpus: List[str] = []
    for conv in dataset:
        raw_chunks = conv.get("chunks", []) or []
        for c in raw_chunks:
            if isinstance(c, str) and c.strip():
                corpus.append(c)
            elif isinstance(c, dict):
                for key in ("chunk_content", "content", "text", "raw", "value"):
                    v = c.get(key)
                    if isinstance(v, str) and v.strip():
                        corpus.append(v)
                        break
    return corpus


def build_dpr_index_over_subchunks(hipporag: HippoRAG, all_subchunks: List[str]) -> None:
    if not all_subchunks:
        return
    logging.info("Inserting %d subchunks into DPR index", len(all_subchunks))
    hipporag.chunk_embedding_store.insert_strings(all_subchunks)
    hipporag.prepare_retrieval_objects()


def build_dpr_index_over_mothers(hipporag: HippoRAG, mothers: List[str]) -> None:
    if not mothers:
        return
    logging.info("Inserting %d mothers into DPR index", len(mothers))
    hipporag.chunk_embedding_store.insert_strings(mothers)
    hipporag.prepare_retrieval_objects()


def aggregate_scores_by_mother(
    docs: List[str],
    scores: List[float],
    sub_to_mother: Dict[str, str],
    top_k: int,
    agg: str = "sum",
) -> Tuple[List[str], List[float]]:
    # Map subchunk docs back to mother chunks; aggregate by sum/mean/max
    bucket: Dict[str, List[float]] = {}
    for d, s in zip(docs, scores):
        mother = sub_to_mother.get(d, d)
        bucket.setdefault(mother, []).append(float(s))

    agg_scores: Dict[str, float] = {}
    for m, arr in bucket.items():
        if agg == "mean":
            agg_scores[m] = sum(arr) / max(1, len(arr))
        elif agg == "max":
            agg_scores[m] = max(arr)
        else:
            agg_scores[m] = sum(arr)

    items = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
    mothers = [m for m, _ in items]
    mscores = [sc for _, sc in items]
    return mothers, mscores


def retrieve_for_questions_with_mapping(
    hipporag: HippoRAG,
    dataset: List[Dict[str, Any]],
    top_k: int,
    sub_to_mother: Dict[str, str],
    pool_factor: int = 10,
    agg: str = "sum",
    use_graph: bool = True,
    debug_path: str | None = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    dbg_f = open(debug_path, "a", encoding="utf-8") if debug_path else None
    for conv in dataset:
        conv_id = conv.get("conv_id")
        conv_chunks = conv.get("chunks", []) or []
        questions = conv.get("questions", []) or []

        for idx, q in enumerate(questions):
            question_text = q.get("question")
            if not isinstance(question_text, str) or not question_text.strip():
                continue

            # Retrieve a larger pool on subchunks, then aggregate back to mothers
            pool_k = max(top_k, int(top_k) * max(1, int(pool_factor)))
            if use_graph:
                # Temporarily set pool size for graph retrieve
                old_k = hipporag.global_config.retrieval_top_k
                hipporag.global_config.retrieval_top_k = pool_k
                sols = hipporag.retrieve(queries=[question_text])
                hipporag.global_config.retrieval_top_k = old_k
            else:
                sols = hipporag.retrieve_dpr(queries=[question_text], num_to_retrieve=pool_k)

            sol = sols[0] if isinstance(sols, list) else sols[0]
            docs = sol.docs or []
            scores = list(sol.doc_scores) if getattr(sol, "doc_scores", None) is not None else []

            # Map subchunks -> mother chunks and aggregate
            mothers, mscores = aggregate_scores_by_mother(docs, scores, sub_to_mother, top_k, agg=agg)
            retrieved_list = [{"chunk": d, "score": float(s)} for d, s in zip(mothers, mscores)]

            if dbg_f:
                json.dump({
                    "type": "retrieval",
                    "conv_id": conv_id,
                    "question_id": q.get("question_id", f"{conv_id or 'conv'}_{idx}"),
                    "query": question_text,
                    "pool_k": pool_k,
                    "agg": agg,
                    "top_subchunks": [{"doc": d, "score": float(s)} for d, s in list(zip(docs, scores))[:10]],
                    "top_mothers": retrieved_list
                }, dbg_f, ensure_ascii=False)
                dbg_f.write("\n")

            out_item: Dict[str, Any] = {
                "conv_id": conv_id,
                "question_id": q.get("question_id", f"{conv_id or 'conv'}_{idx}"),
                "question_type": q.get("question_type", None),
                "question": question_text,
                "evidences": q.get("evidence", q.get("evidences", [])) or [],
                "category": q.get("category"),
                "retrieved": retrieved_list,
                "chunks": conv_chunks,
            }
            results.append(out_item)
    if dbg_f:
        dbg_f.close()
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare HippoRAG DPR retrieval with SeCom segmentation")
    p.add_argument("--input", required=True, help="Path to input dataset JSON (e.g., locomo_session_all.json)")
    p.add_argument("--out", required=True, help="Path to output JSON to save retrieval results")
    p.add_argument("--save-dir", default="outputs/locomo", help="HippoRAG save_dir (embeddings/graph cache)")
    p.add_argument("--llm-name", default="Qwen/Qwen3-8B")
    p.add_argument("--embedding-name", default="BAAI/bge-m3")
    p.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default=None, help="API key for LLM provider (exported to OPENAI_API_KEY)")
    p.add_argument("--embedding-api-key", default=None, help="API key for embedding provider (defaults to --api-key if omitted)")
    p.add_argument("--disable-thinking", action="store_true", help="Disable provider thoughts (e.g., vLLM) via extra_body.disable_thoughts")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--pool-factor", type=int, default=10, help="Multiply top-k on subchunk pool before aggregating to mother")
    p.add_argument("--agg", choices=["sum", "mean", "max"], default="sum")
    p.add_argument("--secom-config", default=None, help="Optional SeCom config.yaml path (defaults to SeCom/secom/configs/mpnet.yaml)")
    p.add_argument("--debug-jsonl", default=None, help="Optional path to write debug JSONL logs")
    p.add_argument("--graph", action="store_true", help="Use full HippoRAG graph (OpenIE) over subchunks before retrieval")
    p.add_argument("--window", action="store_true", help="Augment SeCom with sliding-window subchunks")
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--window-overlap", type=int, default=2)
    p.add_argument("--index-union", action="store_true", help="Index both original chunks and subchunks")
    # Hierarchical retrieval
    p.add_argument("--hierarchical", action="store_true", help="Stage-1 retrieve on mothers, Stage-2 rerank within mothers using subchunks/graph; final returns mothers")
    p.add_argument("--mother-pool", type=int, default=50, help="Candidate mothers to expand at stage-2")
    p.add_argument("--merge-reranker", choices=["none", "bm25"], default="none", help="Optional reranker when merging mothers")
    return p.parse_args()


def main():
    args = parse_args()
    # Quiet INFO-level spam; show warnings and errors only
    logging.basicConfig(level=logging.WARNING)

    dataset = load_dataset(args.input)

    # Allow passing API keys via CLI (export as env for clients)
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.embedding_api_key:
        os.environ.setdefault("OPENAI_API_KEY", args.embedding_api_key)
    # Ensure OpenAI-compatible clients (e.g., SeCom) see the base URL
    if args.llm_base_url:
        os.environ["OPENAI_BASE_URL"] = str(args.llm_base_url)

    original_chunks = collect_original_chunks(dataset)

    # Build SeCom segments and mapping
    mother_to_subs, sub_to_mother = build_secom_segments(
        original_chunks,
        secom_config_path=args.secom_config,
        debug_path=args.debug_jsonl,
        use_window=bool(getattr(args, "window", False)),
        window_size=int(getattr(args, "window_size", 5)),
        window_overlap=int(getattr(args, "window_overlap", 2)),
    )
    # Flatten subchunks for indexing
    all_subchunks: List[str] = [sub for subs in mother_to_subs.values() for sub in subs]
    if bool(getattr(args, "index_union", False)):
        # Also index original chunks
        all_subchunks.extend(original_chunks)

    # Setup HippoRAG config
    config = BaseConfig(
        save_dir=args.save_dir,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        retrieval_top_k=max(1, int(args.top_k)),
        disable_thinking=bool(getattr(args, "disable_thinking", False)),
    )
    hippo = HippoRAG(global_config=config)

    if args.graph:
        # Full Hippo pipeline with OpenIE over subchunks
        hippo.index(all_subchunks)
    else:
        # DPR only over subchunks
        build_dpr_index_over_subchunks(hippo, all_subchunks)

    # Hierarchical or flat retrieval
    if bool(getattr(args, "hierarchical", False)):
        # Mother-only DPR index
        mother_config = BaseConfig(
            save_dir=os.path.join(args.save_dir, "mother_dpr"),
            llm_name=args.llm_name,
            llm_base_url=args.llm_base_url,
            embedding_model_name=args.embedding_name,
            retrieval_top_k=max(1, int(getattr(args, "mother_pool", 50))),
            disable_thinking=bool(getattr(args, "disable_thinking", False)),
        )
        mother_hippo = HippoRAG(global_config=mother_config)
        build_dpr_index_over_mothers(mother_hippo, original_chunks)

        from rank_bm25 import BM25Okapi as _BM25  # lazy import; optional

        results: List[Dict[str, Any]] = []
        dbg_f = open(args.debug_jsonl, "a", encoding="utf-8") if args.debug_jsonl else None
        for conv in dataset:
            conv_id = conv.get("conv_id")
            conv_chunks = conv.get("chunks", []) or []
            questions = conv.get("questions", []) or []
            for idx, q in enumerate(questions):
                question_text = q.get("question")
                if not isinstance(question_text, str) or not question_text.strip():
                    continue

                # Stage-1: mothers
                mother_pool_k = max(1, int(getattr(args, "mother_pool", 50)))
                sols = mother_hippo.retrieve_dpr(queries=[question_text], num_to_retrieve=mother_pool_k)
                sol = sols[0]
                candidate_mothers = sol.docs or []

                # Stage-2: subchunks restricted to candidate mothers
                pool_k = max(config.retrieval_top_k, int(config.retrieval_top_k) * max(1, int(getattr(args, "pool_factor", 10))))
                if bool(getattr(args, "graph", False)):
                    old_k = hippo.global_config.retrieval_top_k
                    hippo.global_config.retrieval_top_k = pool_k
                    sub_sols = hippo.retrieve(queries=[question_text])
                    hippo.global_config.retrieval_top_k = old_k
                else:
                    sub_sols = hippo.retrieve_dpr(queries=[question_text], num_to_retrieve=pool_k)
                sub_sol = sub_sols[0]
                docs = sub_sol.docs or []
                scores = list(sub_sol.doc_scores) if getattr(sub_sol, "doc_scores", None) is not None else []

                # Filter subchunks to candidate mothers only
                cset = set(candidate_mothers)
                f_docs, f_scores = [], []
                for d, s in zip(docs, scores):
                    if sub_to_mother.get(d, d) in cset:
                        f_docs.append(d)
                        f_scores.append(float(s))

                mothers, mscores = aggregate_scores_by_mother(f_docs, f_scores, sub_to_mother, config.retrieval_top_k, agg=str(getattr(args, "agg", "sum")))

                # Optional merge reranker (BM25 over mother texts)
                if str(getattr(args, "merge_reranker", "none")) == "bm25" and mothers:
                    bm = _BM25([m.split() for m in mothers])
                    rerank_scores = bm.get_scores(question_text.split())
                    # Linear combine (simple): normalized sum
                    comb = []
                    for m, a, b in zip(mothers, mscores, rerank_scores):
                        comb.append((m, float(a) + float(b)))
                    comb = sorted(comb, key=lambda x: x[1], reverse=True)[:config.retrieval_top_k]
                    mothers = [m for m, _ in comb]
                    mscores = [s for _, s in comb]

                retrieved_list = [{"chunk": d, "score": float(s)} for d, s in zip(mothers, mscores)]

                if dbg_f:
                    json.dump({
                        "type": "hierarchical",
                        "conv_id": conv_id,
                        "question_id": q.get("question_id", f"{conv_id or 'conv'}_{idx}"),
                        "query": question_text,
                        "mother_pool": candidate_mothers[:10],
                        "top_mothers": retrieved_list,
                    }, dbg_f, ensure_ascii=False)
                    dbg_f.write("\n")

                results.append({
                    "conv_id": conv_id,
                    "question_id": q.get("question_id", f"{conv_id or 'conv'}_{idx}"),
                    "question_type": q.get("question_type", None),
                    "question": question_text,
                    "evidences": q.get("evidence", q.get("evidences", [])) or [],
                    "category": q.get("category"),
                    "retrieved": retrieved_list,
                    "chunks": conv_chunks,
                })
        if dbg_f:
            dbg_f.close()
    else:
        # Retrieve per question, mapping back to original chunks (flat)
        results = retrieve_for_questions_with_mapping(
            hippo,
            dataset,
            top_k=config.retrieval_top_k,
            sub_to_mother=sub_to_mother,
            pool_factor=int(getattr(args, "pool_factor", 10)),
            agg=str(getattr(args, "agg", "sum")),
            use_graph=bool(getattr(args, "graph", False)),
            debug_path=args.debug_jsonl,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote SeCom-segmented retrieval output to: {args.out}  (records={len(results)})")


if __name__ == "__main__":
    main()


"""
python3 /home/hungpv/projects/kustom_hipporag/index_hippo_with_secom_segmentation.py \
  --input /home/hungpv/projects/kustom_hipporag/locomo_session_all.json \
  --out /home/hungpv/projects/kustom_hipporag/locomo_eval_output_secom.json \
  --save-dir outputs/locomo_secom_v1 \
  --llm-base-url http://localhost:8001/v1 \
  --llm-name Qwen/Qwen3-8B \
  --disable-thinking \
  --embedding-name BAAI/bge-m3 \
  --api-key sk-local \
  --top-k 10 \
  --pool-factor 10 \
  --agg sum \
  --graph \
  --window --window-size 5 --window-overlap 2 \
  --index-union \
  --hierarchical --mother-pool 50 --merge-reranker bm25 \
  --debug-jsonl /home/hungpv/projects/kustom_hipporag/secom_debug.jsonl
"""