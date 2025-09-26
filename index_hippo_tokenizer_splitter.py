#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare HippoRAG DPR retrieval outputs using tokenizer-based segmentation.

Flow:
  - For every conversation in the dataset, take its `chunks` (list of strings)
  - Split each chunk by tokens using a Hugging Face tokenizer into subchunks
    with configurable max token length and overlap
  - Build mapping mother_chunk -> [subchunks...] and subchunk -> mother_chunk
  - Index all subchunks into HippoRAG (passage store only) or run full graph if requested
  - For every question, run retrieval over subchunks; aggregate scores back to mother chunks
  - Save output JSON compatible with evaluator.py (retrieved list contains original chunks)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from src.hipporag import HippoRAG  # type: ignore
from src.hipporag.utils.config_utils import BaseConfig  # type: ignore


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _tokenize_split(
    text: str,
    tokenizer_name: str,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    from transformers import AutoTokenizer  # lazy import

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # Encode without truncation to get full token ids
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []

    chunks: List[str] = []
    step = max(1, max_tokens - max(0, overlap_tokens))
    for start in range(0, len(ids), step):
        # Ensure we don't exceed max_tokens
        end = min(start + max_tokens, len(ids))
        window_ids = ids[start:end]
        if not window_ids:
            break
        # Decode the window; skip special tokens just in case
        piece = tokenizer.decode(window_ids, skip_special_tokens=True)
        if isinstance(piece, str) and piece.strip():
            chunks.append(piece)
        if end >= len(ids):
            break
    # Fallback: if decoding leads to nothing, return the original text
    if not chunks:
        return [text]
    return chunks


def build_token_segments(
    chunks: List[str],
    tokenizer_name: str,
    max_tokens: int = 512,
    overlap_tokens: int = 0,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      mother_to_subs: original chunk -> list of subchunks (strings)
      sub_to_mother: subchunk -> original chunk
    """
    mother_to_subs: Dict[str, List[str]] = {}
    sub_to_mother: Dict[str, str] = {}

    for chunk in chunks:
        subs = _tokenize_split(
            chunk,
            tokenizer_name=tokenizer_name,
            max_tokens=max(1, int(max_tokens)),
            overlap_tokens=max(0, int(overlap_tokens)),
        )
        if not subs:
            subs = [chunk]
        mother_to_subs[chunk] = subs
        for s in subs:
            sub_to_mother.setdefault(s, chunk)

    return mother_to_subs, sub_to_mother


def build_dpr_index_over_subchunks(hipporag: HippoRAG, all_subchunks: List[str]) -> None:
    if not all_subchunks:
        return
    logging.info("Inserting %d subchunks into DPR index", len(all_subchunks))
    hipporag.chunk_embedding_store.insert_strings(all_subchunks)
    hipporag.prepare_retrieval_objects()


def aggregate_scores_by_mother(
    docs: List[str],
    scores: List[float],
    sub_to_mother: Dict[str, str],
    top_k: int,
    agg: str = "sum",
) -> Tuple[List[str], List[float]]:
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
    use_graph: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for conv in dataset:
        conv_id = conv.get("conv_id")
        conv_chunks = conv.get("chunks", []) or []
        questions = conv.get("questions", []) or []

        for idx, q in enumerate(questions):
            question_text = q.get("question")
            if not isinstance(question_text, str) or not question_text.strip():
                continue

            pool_k = max(top_k, int(top_k) * max(1, int(pool_factor)))
            if use_graph:
                old_k = hipporag.global_config.retrieval_top_k
                hipporag.global_config.retrieval_top_k = pool_k
                sols = hipporag.retrieve(queries=[question_text])
                hipporag.global_config.retrieval_top_k = old_k
            else:
                sols = hipporag.retrieve_dpr(queries=[question_text], num_to_retrieve=pool_k)

            sol = sols[0]
            docs = sol.docs or []
            scores = list(sol.doc_scores) if getattr(sol, "doc_scores", None) is not None else []

            mothers, mscores = aggregate_scores_by_mother(docs, scores, sub_to_mother, top_k, agg=agg)
            retrieved_list = [{"chunk": d, "score": float(s)} for d, s in zip(mothers, mscores)]

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
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare HippoRAG DPR retrieval with tokenizer-based segmentation")
    p.add_argument("--input", required=True, help="Path to input dataset JSON (e.g., locomo_session_all.json)")
    p.add_argument("--out", required=True, help="Path to output JSON to save retrieval results")
    p.add_argument("--save-dir", default="outputs/locomo", help="HippoRAG save_dir (embeddings/graph cache)")
    p.add_argument("--llm-name", default="Qwen/Qwen3-8B")
    p.add_argument("--embedding-name", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default=None, help="API key for LLM provider (exported to OPENAI_API_KEY)")
    p.add_argument("--embedding-api-key", default=None, help="API key for embedding provider (defaults to --api-key if omitted)")
    p.add_argument("--top-k", type=int, default=10)
    # Tokenizer params
    p.add_argument("--tokenizer-name", default=None, help="HF tokenizer name. Defaults to --embedding-name if omitted")
    p.add_argument("--max-tokens", type=int, default=512, help="Max tokens per subchunk")
    p.add_argument("--overlap-tokens", type=int, default=64, help="Overlap tokens between consecutive subchunks")
    # Retrieval options
    p.add_argument("--pool-factor", type=int, default=10, help="Multiply top-k on subchunk pool before aggregating to mother")
    p.add_argument("--agg", choices=["sum", "mean", "max"], default="sum")
    p.add_argument("--graph", action="store_true", help="Use full HippoRAG graph (OpenIE) over subchunks before retrieval")
    p.add_argument("--index-union", action="store_true", help="Also index original chunks together with subchunks")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset = load_dataset(args.input)

    # Allow passing API keys via CLI (export as env for clients)
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.embedding_api_key:
        os.environ.setdefault("OPENAI_API_KEY", args.embedding_api_key)
    if args.llm_base_url:
        os.environ["OPENAI_BASE_URL"] = str(args.llm_base_url)

    original_chunks = collect_original_chunks(dataset)

    # Tokenizer defaults to embedding name when not provided
    tokenizer_name = (
        args.tokenizer_name if isinstance(args.tokenizer_name, str) and args.tokenizer_name.strip() else args.embedding_name
    )

    mother_to_subs, sub_to_mother = build_token_segments(
        original_chunks,
        tokenizer_name=tokenizer_name,
        max_tokens=int(getattr(args, "max_tokens", 512)),
        overlap_tokens=int(getattr(args, "overlap_tokens", 64)),
    )
    all_subchunks: List[str] = [sub for subs in mother_to_subs.values() for sub in subs]
    if bool(getattr(args, "index_union", False)):
        all_subchunks.extend(original_chunks)

    config = BaseConfig(
        save_dir=args.save_dir,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        retrieval_top_k=max(1, int(args.top_k)),
    )
    hippo = HippoRAG(global_config=config)

    if bool(getattr(args, "graph", False)):
        hippo.index(all_subchunks)
    else:
        build_dpr_index_over_subchunks(hippo, all_subchunks)

    results = retrieve_for_questions_with_mapping(
        hippo,
        dataset,
        top_k=config.retrieval_top_k,
        sub_to_mother=sub_to_mother,
        pool_factor=int(getattr(args, "pool_factor", 10)),
        agg=str(getattr(args, "agg", "sum")),
        use_graph=bool(getattr(args, "graph", False)),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote tokenizer-segmented retrieval output to: {args.out}  (records={len(results)})")


if __name__ == "__main__":
    main()

"""
Example:
python3 /home/hungpv/projects/kustom_hipporag/index_hippo_tokenizer_splitter.py \
  --input /home/hungpv/projects/kustom_hipporag/locomo_session_all.json \
  --out /home/hungpv/projects/kustom_hipporag/locomo_eval_output_tokenizer.json \
  --save-dir outputs/locomo_tokenizer \
  --llm-base-url http://localhost:8001/v1 \
  --llm-name Qwen/Qwen3-8B \
  --embedding-name sentence-transformers/multi-qa-mpnet-base-dot-v1 \
  --api-key sk-local \
  --top-k 10 \
  --tokenizer-name sentence-transformers/multi-qa-mpnet-base-dot-v1 \
  --max-tokens 512 --overlap-tokens 64 \
  --pool-factor 10 --agg sum
"""


