from typing import List
import json

import torch
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction
from sentence_transformers import SentenceTransformer

class TransformersEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "Transformers/"
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        # Accept both prefixed ("Transformers/<repo>") and bare HF repo IDs (e.g., "BAAI/bge-m3")
        if embedding_model_name.startswith("Transformers/"):
            self.model_id = embedding_model_name[len("Transformers/"):]
        else:
            self.model_id = embedding_model_name
        self.embedding_type = 'float'
        # use global config for batch size if set
        try:
            self.batch_size = int(getattr(self.global_config, 'embedding_batch_size', 64)) or 64
        except Exception:
            self.batch_size = 64

        # Prefer CUDA, then Apple Silicon MPS, else CPU
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"

        self.model = SentenceTransformer(self.model_id, device=device)

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def encode(self, texts: List[str]) -> None:
        try:
            response = self.model.encode(texts, batch_size=self.batch_size)
        except Exception as err:
            raise Exception(f"An error occurred: {err}")
        return np.array(response)

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if len(texts) < self.batch_size:
            return self.encode(texts)
        
        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        disable_bar = len(batch_indexes) <= 1
        for i in tqdm(batch_indexes, desc="Batch Encoding", disable=disable_bar, mininterval=0.5):
            results.append(self.encode(texts[i:i + self.batch_size]))
        return np.concatenate(results)
