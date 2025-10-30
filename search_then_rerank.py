# SEARCH WITH RETRIEVER THEN RERANK
import os, json, argparse, faiss, numpy as np, torch
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

from utils import stable64

from search_with_retriever import get_chunks_by_hash


class Embedder:
    def __init__(self, model_id: str, dtype=DTYPE, device=DEVICE):
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.enc = (
            AutoModel.from_pretrained(model_id, torch_dtype=dtype).eval().to(device)
        )
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def embed_texts(
        self, texts: List[str], batch_size=64, max_length=512
    ) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc_in = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc_in["input_ids"].to(self.device, non_blocking=True)
            attn = enc_in["attention_mask"].to(self.device, non_blocking=True)
            with (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if self.device == "cuda"
                else torch.no_grad()
            ):
                out = self.enc(input_ids=input_ids, attention_mask=attn)
                last = out.last_hidden_state
                mask = attn.unsqueeze(-1)
                emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.append(emb.float().cpu().numpy())
        return np.vstack(vecs)


def load_meta_map(meta_path: str) -> Dict[int, dict]:
    id2meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            md = json.loads(line)
            iid = int(stable64(md["chunk_id"]))
            id2meta[iid] = md
    return id2meta


def apply_boosts(
    sim: np.ndarray, ids: np.ndarray, id2meta: Dict[int, dict], mode="mul", lam=0.1
) -> np.ndarray:
    boosts = np.array(
        [id2meta.get(int(i), {}).get("boost", 1.0) for i in ids], dtype=np.float32
    )
    if mode == "mul":
        return sim * boosts
    return sim + lam * np.log(np.clip(boosts, 1e-6, None))


def cap_per_paper(
    ids_sorted: List[int],
    scores_sorted: List[float],
    id2meta: Dict[int, dict],
    per_paper=2,
    keep=50,
) -> List[Tuple[int, float]]:
    per_count = defaultdict(int)
    taken = []
    for iid, sc in zip(ids_sorted, scores_sorted):
        md = id2meta.get(int(iid), {})
        pid = md.get("paper_id")
        if pid is None:
            continue
        if per_count[pid] >= per_paper:
            continue
        per_count[pid] += 1
        taken.append((int(iid), float(sc)))
        if len(taken) >= keep:
            break
    return taken


class Reranker:
    def __init__(self, model_id="OpenScholar/OpenScholar_Reranker"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_id)
            .eval()
            .to(DEVICE)
        )

    @torch.inference_mode()
    def score(
        self, query: str, passages: List[str], batch_size=32, max_len=512
    ) -> List[float]:
        scores = []
        for i in range(0, len(passages), batch_size):
            pairs = [
                [query, passages[j]]
                for j in range(i, min(i + batch_size, len(passages)))
            ]
            enc = self.tok(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(DEVICE)
            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(
                logits.detach().cpu().tolist()
                if logits.ndim
                else [float(logits.detach().cpu())]
            )
        return scores


def make_header(md: dict) -> str:
    title = md.get("title", "")
    section = (md.get("section") or "").upper()
    subsection = md.get("subsection")
    hdr = f"{title} — {section}" if title or section else ""
    if subsection:
        hdr += f" / {subsection}"
    return hdr


def run(args):
    index = faiss.read_index(args.index_path)
    id2meta = load_meta_map(args.meta_path)

    retriever = Embedder(args.retriever_model)
    q_vec = retriever.embed_texts(
        [args.query], batch_size=1, max_length=args.max_length
    ).astype(np.float32)

    sims, ids = index.search(q_vec, args.initial_topk)  # (1,k)
    sims, ids = sims[0], ids[0]

    resc = apply_boosts(sims, ids, id2meta, mode=args.boost_mode, lam=args.boost_lambda)
    order = np.argsort(-resc)
    ids_sorted = ids[order]
    scores_sorted = resc[order]

    kept = cap_per_paper(
        ids_sorted.tolist(),
        scores_sorted.tolist(),
        id2meta,
        per_paper=args.per_paper_cap,
        keep=args.keep_for_rerank,
    )

    # these are FAISS IDs
    kept_hash_ids = [iid for iid, _ in kept]
    chunks_df = get_chunks_by_hash(kept_hash_ids)

    cid_to_text = {}
    for _, row in chunks_df.iterrows():
        cid_to_text[row["chunk_id"]] = row["text"]

    passages, meta_list, dois = [], [], []
    for iid, pre_sc in kept:
        md = id2meta[iid]
        cid = md["chunk_id"]
        body = cid_to_text.get(cid, "")
        header = make_header(md)
        passage = f"{header}\n\n{body}" if header else body
        passages.append(passage)
        dois.append(md["paper_id"])
        meta_list.append((iid, pre_sc, md))

    reranker = Reranker(args.reranker_model)
    ce_scores = reranker.score(args.query, passages, batch_size=32, max_len=512)

    order2 = np.argsort(-np.array(ce_scores))
    print("\nTop reranked passages:\n")
    for r in order2[: args.final_topk]:
        score = f"{ce_scores[r]:.4f}"
        doi = dois[r]
        passage = passages[r]

        print(f"Reranker score: {score}")
        print(f"DOI: {doi}")
        print(f"Passage: {passage}\n{'-'*60}\n")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--retriever_model", default="OpenSciLM/OpenScholar_Retriever")
    ap.add_argument("--reranker_model", default="OpenScholar/OpenScholar_Reranker")
    ap.add_argument("--query", required=True)
    ap.add_argument("--initial_topk", type=int, default=400)
    ap.add_argument("--keep_for_rerank", type=int, default=80)
    ap.add_argument("--final_topk", type=int, default=10)
    ap.add_argument("--per_paper_cap", type=int, default=3)
    ap.add_argument("--boost_mode", choices=["mul", "add"], default="mul")
    ap.add_argument("--boost_lambda", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()
    run(args)
