# BUILD A PARQUET FOR EFFICIENT AND FAST QUERYING
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import pyarrow.parquet as pq

from build_index import (
    discover_inputs,
    load_json_records,
    make_chunks,
    Embedder,
    stable64,
)
import pathlib


def build_chunks_database_from_meta(index_dir: str, data_dir: str, output_path: str):
    """Build parquet file using the meta.jsonl and reconstructing texts."""

    meta_path = os.path.join(index_dir, "meta.jsonl")

    print("Loading metadata...")
    meta_records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta_records.append(json.loads(line))

    print(f"Loaded {len(meta_records)} metadata records")

    # crreate a mapping of chunk_id to text by re-processing the data files
    print("Rebuilding chunk texts...")
    chunk_id_to_text = {}

    embedder = Embedder("OpenSciLM/OpenScholar_Retriever")
    tok = embedder.tok

    files = discover_inputs(data_dir)

    for fp in tqdm(files, desc="Processing files"):
        for paper in load_json_records(fp):
            paper_id = paper["doi"]
            for chunk_id, text, md in make_chunks(paper, paper_id, tok):
                chunk_id_to_text[chunk_id] = text

    print("Combining metadata with texts...")
    chunks_data = []
    missing_texts = 0

    for meta in tqdm(meta_records, desc="Building final dataset"):
        chunk_id = meta["chunk_id"]
        text = chunk_id_to_text.get(chunk_id, "")

        if not text:
            missing_texts += 1
            continue

        chunk_data = {
            "chunk_id": chunk_id,
            "chunk_hash": stable64(chunk_id),
            "text": text,
            "paper_id": meta.get("paper_id", ""),
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "subsection": meta.get("subsection", ""),
            "paragraph_index": meta.get("paragraph_index", 0),
            "keywords": json.dumps(meta.get("keywords", [])),
            "boost": meta.get("boost", 1.0),
        }
        chunks_data.append(chunk_data)

    print(f"Missing texts: {missing_texts}")

    df = pd.DataFrame(chunks_data)

    df.to_parquet(output_path, compression="snappy", engine="pyarrow")

    print(f"Saved {len(df)} chunks to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")

    return df


# alternatively build directly
def build_chunks_database_direct(data_dir: str, output_path: str):
    """Build parquet file directly from data files."""

    chunks_data = []

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "OpenSciLM/OpenScholar_Retriever", use_fast=True
    )

    files = discover_inputs(data_dir)

    for fp in tqdm(files, desc="Processing files"):
        for paper in load_json_records(fp):
            paper_id = paper["doi"]
            for chunk_id, text, meta in make_chunks(paper, paper_id, tok):
                chunk_data = {
                    "chunk_id": chunk_id,
                    "chunk_hash": stable64(chunk_id),
                    "text": text,
                    "paper_id": meta["paper_id"],
                    "title": meta["title"],
                    "section": meta["section"],
                    "subsection": meta["subsection"] or "",
                    "paragraph_index": meta["paragraph_index"],
                    "keywords": json.dumps(meta["keywords"]),
                    "boost": meta["boost"],
                }
                chunks_data.append(chunk_data)

    df = pd.DataFrame(chunks_data)

    df.to_parquet(output_path, compression="snappy", engine="pyarrow")

    print(f"Saved {len(df)} chunks to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="bio-openscholar/data")
    parser.add_argument(
        "--index_dir",
        type=str,
        default="./bio-openscholar/index_1",
        help="Directory with existing index and meta.jsonl",
    )
    parser.add_argument("--output", type=str, default="openscholar_chunks.parquet")
    parser.add_argument(
        "--method", type=str, choices=["from_meta", "direct"], default="from_meta"
    )

    args = parser.parse_args()

    if args.method == "from_meta":
        df = build_chunks_database_from_meta(args.index_dir, args.data_dir, args.output)
    else:
        df = build_chunks_database_direct(args.data_dir, args.output)
