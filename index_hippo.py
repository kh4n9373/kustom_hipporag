#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare HippoRAG DPR retrieval outputs for evaluator.py

Inputs:
  - --input DATASET_JSON (e.g., locomo_session_all.json)
  - --out OUTPUT_JSON path
  - Optional model/config flags

Behavior:
  - Aggregate all conversation-level `chunks` as the corpus
  - Build HippoRAG embedding storage by directly inserting chunks (skip OpenIE)
  - For every question in every conversation, run DPR retrieval
  - Save a JSON array; each item contains:
      conv_id, question_id (if present or synthesized), question_type (if present),
      question, evidences, category, retrieved (list[{chunk, score}]), chunks (source conv chunks)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List
from src.hipporag import HippoRAG  # type: ignore
from src.hipporag.utils.config_utils import BaseConfig  # type: ignore


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_corpus(dataset: List[Dict[str, Any]]) -> List[str]:
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


def build_dpr_index(hipporag: HippoRAG, corpus: List[str]) -> None:
    if not corpus:
        return
    logging.info("Inserting %d chunks into DPR index", len(corpus))
    hipporag.chunk_embedding_store.insert_strings(corpus)
    hipporag.prepare_retrieval_objects()


def retrieve_for_questions(hipporag: HippoRAG,
                           dataset: List[Dict[str, Any]],
                           top_k: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for conv in dataset:
        conv_id = conv.get("conv_id")
        conv_chunks = conv.get("chunks", []) or []
        questions = conv.get("questions", []) or []

        for idx, q in enumerate(questions):
            question_text = q.get("question")
            if not isinstance(question_text, str) or not question_text.strip():
                continue

            retrieved = hipporag.retrieve_dpr(queries=[question_text], num_to_retrieve=top_k)
            sol = retrieved[0]
            docs = sol.docs or []
            scores = list(sol.doc_scores) if sol.doc_scores is not None else []
            retrieved_list = []
            for d, s in zip(docs, scores):
                if isinstance(d, str):
                    retrieved_list.append({"chunk": d, "score": float(s)})

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
    p = argparse.ArgumentParser(description="Prepare HippoRAG DPR retrieval output for evaluator")
    p.add_argument("--input", required=True, help="Path to input dataset JSON (e.g., locomo_session_all.json)")
    p.add_argument("--out", required=True, help="Path to output JSON to save retrieval results")
    p.add_argument("--save-dir", default="outputs/locomo", help="HippoRAG save_dir (embeddings/graph cache)")
    p.add_argument("--llm-name", default=None, help="LLM model name (env: LLM_NAME)")
    p.add_argument("-m", "--model", dest="model", default=None, help="Alias for --llm-name")
    p.add_argument("--embedding-name", default="BAAI/bge-m3")
    p.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default=None, help="API key for LLM provider (exported to OPENAI_API_KEY)")
    p.add_argument("--embedding-api-key", default=None, help="API key for embedding provider (defaults to --api-key if omitted)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--disable-thinking", action="store_true", help="No-op flag for compatibility; ignored")
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

    corpus = collect_corpus(dataset)

    # Resolve LLM name from flags/env with sensible defaults
    resolved_llm_name = (
        (args.llm_name if isinstance(args.llm_name, str) and args.llm_name.strip() else None)
        or (args.model if isinstance(args.model, str) and args.model.strip() else None)
        or os.environ.get("LLM_NAME")
        or "Qwen/Qwen3-8B"
    )

    # Minimal config; we'll use only DPR paths, so OpenIE/graph are not required
    config = BaseConfig(
        save_dir=args.save_dir,
        llm_name=resolved_llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        retrieval_top_k=max(1, int(args.top_k)),
    )

    hippo = HippoRAG(global_config=config)

    # Build DPR index from chunks only
    build_dpr_index(hippo, corpus)

    # Retrieve per question
    results = retrieve_for_questions(hippo, dataset, top_k=config.retrieval_top_k)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote retrieval output to: {args.out}  (records={len(results)})")


if __name__ == "__main__":
    main()

"""
python3 index_hippo.py \
  --input /home/hungpv/projects/kustom_hipporag/locomo_session_all.json \
  --out /home/hungpv/projects/kustom_hipporag/locomo_eval_output.json \
  --save-dir outputs/locomo_secom \
  --llm-base-url http://localhost:8002/v1 \
  --llm-name Qwen/Qwen3-8B \
  --disable-thinking \
  --embedding-name BAAI/bge-m3 \
  --api-key sk- \
  --top-k 10 \
"""