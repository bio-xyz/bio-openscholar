# TESTS ONLY USING RETREIVER
import os
import json
import faiss
import numpy as np
import pandas as pd
import torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel

from utils import stable64

INDEX_DIR = "./bio-openscholar/index_1"
CHUNKS_PARQUET = "./bio-openscholar/index_1/data/train.parquet"
MODEL_ID = "OpenSciLM/OpenScholar_Retriever"

print("Loading chunks database...")
chunks_df = pd.read_parquet(CHUNKS_PARQUET)

hash_to_chunk_id = dict(zip(chunks_df["chunk_hash"], chunks_df["chunk_id"]))
chunk_id_to_row = dict(zip(chunks_df["chunk_id"], chunks_df.index))

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
enc = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval().to("cuda")


@torch.inference_mode()
def embed_query(q: str, max_len: int = 512) -> np.ndarray:
    enc_in = tok(
        q, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    ).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = enc(**enc_in)
        last = out.last_hidden_state
        mask = enc_in["attention_mask"].unsqueeze(-1)
        v = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        v = torch.nn.functional.normalize(v, p=2, dim=1)
    return v.float().cpu().numpy()


def get_chunks_by_hash(chunk_hashes: List[int]) -> pd.DataFrame:
    """Get chunk data by hash values."""
    chunk_ids = [hash_to_chunk_id.get(h) for h in chunk_hashes if h in hash_to_chunk_id]
    indices = [chunk_id_to_row[cid] for cid in chunk_ids if cid in chunk_id_to_row]
    return chunks_df.iloc[indices]


if __name__ == "__main__":
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))

    id_to_meta: Dict[int, dict] = {}
    with open(os.path.join(INDEX_DIR, "meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            md = json.loads(line)
            iid = int(stable64(md["chunk_id"]))
            id_to_meta[iid] = md

    query = 'Effects of Immunization With the Soil-Derived Bacterium Mycobacterium vaccae on Stress Coping Behaviors and Cognitive Performance in a "Two Hit" Stressor Model'
    qv = embed_query(query)
    D, I = index.search(qv, k=100)

    chunk_hashes = I[0].tolist()
    results_df = get_chunks_by_hash(chunk_hashes)

    print(f"\nQUERY: {query}")
    print(f"Found {len(results_df)} results\n")

    for rank, (score, hash_id) in enumerate(zip(D[0], I[0]), 1):
        if hash_id in hash_to_chunk_id:
            chunk_id = hash_to_chunk_id[hash_id]
            row = chunks_df[chunks_df["chunk_id"] == chunk_id].iloc[0]

            text = row["text"]
            preview = (text[:240] + "…") if len(text) > 260 else text

            print(f"#{rank}  score={score:.3f}")
            print(f"  paper_id: {row['paper_id']}")
            print(f"  section : {row['section']}  | subsection: {row['subsection']}")
            print(f"  para_idx: {row['paragraph_index']}  | boost: {row['boost']}")
            print(f"  preview : {preview}")
            print()
