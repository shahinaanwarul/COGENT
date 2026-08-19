#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd

from cogent.data import normalize_hashtag
from cogent.semantic import HashingSemanticEncoder, LlamaSemanticEncoder, save_embedding_map


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute hashtag embeddings used for topics and semantic tokens.")
    ap.add_argument("--input", required=True, help="CSV with a Hashtag column")
    ap.add_argument("--output", required=True, help="Output .npz")
    ap.add_argument("--encoder", choices=["llama3", "hashing"], default="llama3")
    ap.add_argument("--hash-dim", type=int, default=256)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    hashtags = sorted({normalize_hashtag(x) for x in df["Hashtag"].dropna().astype(str)})
    if args.encoder == "llama3":
        enc = LlamaSemanticEncoder()
    else:
        enc = HashingSemanticEncoder(args.hash_dim)
    emb = enc.encode(hashtags)
    save_embedding_map(args.output, hashtags, emb)
    print(f"Saved {len(hashtags)} embeddings of dimension {emb.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
