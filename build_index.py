import os, re, json, glob, math, argparse, faiss, numpy as np, torch, pathlib
from typing import Dict, List, Iterable, Tuple
from transformers import AutoTokenizer, AutoModel
from utils import stable64
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = "cuda"
DTYPE = torch.bfloat16

MAX_CTX = 512
CHUNK_TOKS = min(380, MAX_CTX)  # safe cap
OVERLAP = int(CHUNK_TOKS * 0.15)  # ~15%
MIN_TOKS = int(CHUNK_TOKS * 0.5)  # don't make tiny shards

SECTION_ORDER = [
    "title",
    "abstract",
    "introduction",
    "results",
    "discussion",
    "conclusion",
    "methods",
    "supplementary",
]


def iter_sections(paper: Dict):
    for key in SECTION_ORDER:
        if key in paper and paper[key]:
            text = paper[key]
            if key == "methods":
                subs = re.split(r"\n\n(?=[A-Z][^\n]{3,80}\n)", text)
                for s in subs:
                    lines = s.strip().split("\n", 1)
                    subname = lines[0].strip() if len(lines) > 1 else key
                    body = lines[1] if len(lines) > 1 else s
                    yield key, subname, body
            else:
                yield key, None, text


def paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    merged, buf, words = [], [], 0
    for p in paras:
        w = len(p.split())
        buf.append(p)
        words += w
        if words >= 120:  # ~120 words per merged paragraph
            merged.append("\n\n".join(buf))
            buf, words = [], 0
    if buf:
        merged.append("\n\n".join(buf))
    return merged


def sentence_chunks(text: str, tok, max_toks=320, min_toks=160, sent_overlap=2):
    """
    Pack whole sentences up to ~max_toks using the retriever tokenizer.
    Keep a couple of overlapping sentences between chunks.
    """
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    i = 0
    while i < len(sents):
        buf = []
        tok_count = 0
        j = i
        while j < len(sents):
            tentative = (" ".join(buf + [sents[j]])).strip()
            n_toks = len(tok.encode(tentative, add_special_tokens=False))
            if tok_count >= min_toks and n_toks > max_toks:
                break
            if n_toks > max_toks and not buf:
                # snigle long sentence: fall back to token slice
                ids = tok.encode(sents[j], add_special_tokens=False)
                chunk_ids = ids[:max_toks]
                yield tok.decode(chunk_ids)
                j += 1
                break
            buf.append(sents[j])
            tok_count = n_toks
            j += 1
        if buf:
            yield " ".join(buf)
        if j >= len(sents):
            break
        # sentence-level overlap
        i = max(i + 1, j - sent_overlap)


def token_chunks(
    text: str, tok, max_toks=CHUNK_TOKS, overlap=OVERLAP, min_toks=MIN_TOKS
):
    ids = tok.encode(text, add_special_tokens=False)
    i, n = 0, len(ids)
    while i < n:
        j = min(i + max_toks, n)
        if j - i < min_toks and j < n:
            j = min(n, i + min_toks)
        chunk_ids = ids[i:j]
        yield tok.decode(chunk_ids)
        if j == n:
            break
        i = max(0, j - overlap)


def make_chunks(paper: Dict, paper_id: str, tok) -> Iterable[Tuple[str, str, Dict]]:
    """
    Yields (chunk_id, text, metadata)
    """
    try:
        title = paper.get("title", "")
    except:
        print(paper_id)
        print(paper)
        paper = json.loads(paper)
        title = paper.get("title", "")
    keywords = paper.get("keywords", [])
    for section, subsection, body in iter_sections(paper):
        for para_idx, para in enumerate(paragraphs(body)):
            frag_idx = 0
            for frag in sentence_chunks(
                para, tok, max_toks=320, min_toks=160, sent_overlap=2
            ):
                chunk_id = f"{paper_id}:::{section}:::{subsection or ''}:::{para_idx}:::{frag_idx}"
                header = f"{title} — {section.upper()}"
                if subsection:
                    header += f" / {subsection}"
                frag_with_hdr = f"{header}\n\n{frag}"  # Adds context (title + section header) to each chunk for better querying
                md = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title": title,
                    "section": section,
                    "subsection": subsection,
                    "paragraph_index": para_idx,
                    "keywords": keywords,
                    "boost": (
                        1.3
                        if section in {"abstract", "results", "conclusion"}
                        else (0.9 if section == "methods" else 1.0)
                    ),
                }
                yield chunk_id, frag_with_hdr, md
                frag_idx += 1


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
        self, texts: List[str], batch_size=128, max_length=512
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
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.enc(input_ids=input_ids, attention_mask=attn)
                last = out.last_hidden_state  # [B, T, H]
                mask = attn.unsqueeze(-1)  # [B, T, 1]
                summed = (last * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                emb = summed / denom
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.append(emb.float().cpu().numpy())
        return np.vstack(vecs)


def make_index(d: int, kind: str):
    if kind == "flat":
        return faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 128
        return idx
    elif kind.startswith("ivf"):
        # e.g., ivf16384, ivf8192
        nlist = int(kind.replace("ivf", ""))
        quant = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT
        )
        quant.nprobe = max(8, int(nlist * 0.01))
        return quant
    else:
        raise ValueError(f"Unknown index kind: {kind}")


def load_json_records(path: str) -> Iterable[Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                for rec in obj:
                    yield rec
            else:  # single paper
                yield obj
    else:
        return


def discover_inputs(data_dir: str) -> List[str]:
    files = []
    files += glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
    files += glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    files = [p for p in files if not os.path.basename(p).startswith(".")]
    return sorted(set(files))


# main
def build(args):
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    done_path = os.path.join(args.out_dir, "done_chunk_ids.txt")
    index_path = os.path.join(args.out_dir, "index.faiss")
    dim = None

    # resume support, yay
    done = set()
    if os.path.exists(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            for line in f:
                done.add(line.strip())

    embedder = Embedder(args.model)
    tok = embedder.tok

    # temporary buffer before adding to FAISS (to avoid tiny add calls)
    buf_texts, buf_meta, buf_ids = [], [], []

    meta_out = open(meta_path, "a", encoding="utf-8")

    index = None

    files = discover_inputs(args.data_dir)
    if not files:
        raise SystemExit(f"No JSON/JSONL found under: {args.data_dir}")
    else:
        print(f"Loaded {len(files)} JSONs")

    for fp in files:
        for paper in load_json_records(fp):
            paper_id = paper["doi"]
            for chunk_id, text, md in make_chunks(paper, paper_id, tok):
                if chunk_id in done:
                    continue
                buf_texts.append(text)
                buf_meta.append(md)
                buf_ids.append(chunk_id)

                if len(buf_texts) >= args.embed_batch * 4:
                    embs = embedder.embed_texts(
                        buf_texts,
                        batch_size=args.embed_batch,
                        max_length=args.max_length,
                    )
                    if dim is None:
                        dim = embs.shape[1]
                        index = make_index(dim, args.index_kind)
                        # if IVF, need to train
                        if isinstance(index, faiss.IndexIVF):
                            # sample subset to train
                            train_sample = embs[
                                np.random.choice(
                                    len(embs), size=min(50000, len(embs)), replace=False
                                )
                            ]
                            index.train(train_sample)

                    if not isinstance(index, faiss.IndexIDMap2):
                        index = faiss.IndexIDMap2(index)

                    int_ids = np.fromiter(
                        (stable64(cid) for cid in buf_ids),
                        dtype="int64",
                        count=len(buf_ids),
                    )
                    index.add_with_ids(embs, int_ids)

                    for md in buf_meta:
                        meta_out.write(json.dumps(md, ensure_ascii=False) + "\n")
                    meta_out.flush()
                    with open(done_path, "a", encoding="utf-8") as f:
                        for cid in buf_ids:
                            f.write(cid + "\n")

                    buf_texts, buf_meta, buf_ids = [], [], []

    if buf_texts:
        embs = embedder.embed_texts(
            buf_texts, batch_size=args.embed_batch, max_length=args.max_length
        )
        if dim is None:
            dim = embs.shape[1]
            index = make_index(dim, args.index_kind)
            if isinstance(index, faiss.IndexIVF):
                train_sample = embs[
                    np.random.choice(
                        len(embs), size=min(50000, len(embs)), replace=False
                    )
                ]
                index.train(train_sample)
        # boost = np.array([m["boost"] for m in buf_meta], dtype=np.float32).reshape(
        #     -1, 1
        # )
        # embs = embs * boost
        if not isinstance(index, faiss.IndexIDMap2):
            index = faiss.IndexIDMap2(index)
        int_ids = np.fromiter(
            (stable64(cid) for cid in buf_ids), dtype="int64", count=len(buf_ids)
        )
        index.add_with_ids(embs, int_ids)
        for md in buf_meta:
            meta_out.write(json.dumps(md, ensure_ascii=False) + "\n")
        meta_out.flush()
        with open(done_path, "a", encoding="utf-8") as f:
            for cid in buf_ids:
                f.write(cid + "\n")

    meta_out.close()

    faiss.write_index(index, index_path)
    print(f"Saved index to {index_path}")
    print(f"Wrote meta to {meta_path}")
    print("Done.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with .json or .jsonl files (one paper or list of papers).",
    )
    ap.add_argument(
        "--out_dir", type=str, required=True, help="Output dir for index and meta."
    )
    ap.add_argument("--model", type=str, default="OpenSciLM/OpenScholar_Retriever")
    ap.add_argument(
        "--index_kind",
        type=str,
        default="flat",
        choices=["flat", "hnsw", "ivf16384", "ivf8192"],
    )
    ap.add_argument("--embed_batch", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=512)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args)

"""python bio-openscholar/build_index.py \
  --data_dir bio-openscholar/data \
  --out_dir bio-openscholar/index_1"""
