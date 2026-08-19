from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
import json

import numpy as np


class SemanticEncoder:
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HashingSemanticEncoder(SemanticEncoder):
    """Deterministic lightweight encoder for smoke tests and CPU-only runs.

    It is not the encoder used in the manuscript. The manuscript specifies the
    frozen meta-llama/Meta-Llama-3-8B-Instruct final-layer embedding with masked
    mean pooling. Use LlamaSemanticEncoder for manuscript-faithful semantic features.
    """

    dimension: int = 256

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer

        vec = HashingVectorizer(
            n_features=self.dimension,
            alternate_sign=False,
            norm="l2",
            analyzer="char_wb",
            ngram_range=(3, 5),
        )
        return vec.transform(list(texts)).toarray().astype(np.float32)


class LlamaSemanticEncoder(SemanticEncoder):
    """Frozen LLaMA-3 embedding encoder described in the manuscript.

    The class lazily imports Hugging Face transformers so the rest of the package
    can run without that optional dependency. Access to the Meta Llama checkpoint
    may require accepting the model license and authenticating with Hugging Face.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        max_length: int = 128,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LlamaSemanticEncoder requires `transformers` and `accelerate`. "
                "Install the optional dependencies in requirements-full.txt."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"output_hidden_states": True}
        if device == "auto":
            kwargs["device_map"] = "auto"
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_length = max_length
        self.dimension = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str], batch_size: int = 4) -> np.ndarray:
        torch = self._torch
        outputs: List[np.ndarray] = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                toks = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                out = self.model(**toks)
                h = out.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                outputs.append(pooled.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def save_embedding_map(path: str | Path, hashtags: Sequence[str], embeddings: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, hashtags=np.asarray(hashtags, dtype=object), embeddings=np.asarray(embeddings, dtype=np.float32))


def load_embedding_map(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    hashtags = data["hashtags"].tolist()
    emb = data["embeddings"]
    return {str(h): emb[i].astype(np.float32) for i, h in enumerate(hashtags)}
