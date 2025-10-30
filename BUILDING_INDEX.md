# Building FAISS index to use with OpenScholar base models

This directory contains the scripts to convert your cleaned raw manuscripts into an optmizied FAISS index for efficient querying during runtime usning base models used in openscholar.

## Prerequisites

- A consistently structured source archive: You will get this if you extract the raw text using [MinerU](https://github.com/opendatalab/MinerU). Execute this command to get the folder structure.

```bash
mineru -p <input_path_to_pdfs> -o <output_path_dir>
```

After running this command each paper should have its own folder with a manuscript file plus a `vlm/images` subdirectory for figures. Eg. below.

<img width="441" height="314" alt="image" src="https://github.com/user-attachments/assets/4f201907-f5f7-4e18-95d0-bd2a36177431" />

- GROBID for getting paper header info like DOI, authors, etc.
- Shared [prompt](/prompts.py) definitions and helper utilities bundled here for image narration, document cleaning, structured extraction, and response parsing.

## Data cleaning flow (High level overview)

You can use any data-cleaning flow that fits your stack, but ensure the cleaned corpus meets these expectations:

- Strip noise such as page numbers, headers/footers, and author lists so only manuscript prose remains.
- Replace images with short, human-written descriptions to preserve multimodal cues.
- Produce one JSON object per paper with the fields below, filling empty sections with `""`:

```json
{
  "doi": "",
  "title": "",
  "abstract": "",
  "introduction": "",
  "results": "",
  "discussion": "",
  "conclusion": "",
  "methods": "",
  "supplementary": "",
  "keywords": []
}
```

An example file is here for your reference: [structured_with_doi.jsonl](/data/structured_with_doi.jsonl)

Look up DOIs with GROBID, an automated search, or a retrieval-augmented LLM. When every paper has this shape, concatenate them into a single JSONL file that downstream steps can stream.

## Build FAISS index:

```bash
python bio-openscholar/build_index.py \
  --data_dir bio-openscholar/data \
  --out_dir bio-openscholar/index_1
```

Point `--data_dir` to the directory that holds your cleaned JSONL. The output folder will contain `meta.jsonl` and `index.faiss`.

`meta.jsonl` is the metadata sidecar for the FAISS index. Each line maps an embedding ID to the paper context, making it possible to recover provenance during search. Typical fields include the parent DOI, section name, and keywords:

```json
{
  "chunk_id": "10.22074/ijfs.2018.5185:::title::::::0:::0",
  "paper_id": "10.22074/ijfs.2018.5185",
  "title": "Relationship between Health Literacy and Sexual Function and Sexual Satisfaction in Infertile Couples Referred to The Royan Institute",
  "section": "title",
  "subsection": null,
  "paragraph_index": 0,
  "keywords": [
    "Health Literacy",
    "Infertility",
    "Sexual Dysfunction",
    "Sexual Satisfaction"
  ],
  "boost": 1.0
}
```

Note: the chunk body text itself is not stored in `meta.jsonl`; generate a companion parquet (next step) so search-time pipelines can hydrate passages.

```bash
python bio-openscholar/build_parquet.py \
  --data_dir bio-openscholar/data \
  --index_dir bio-openscholar/index_1 \
  --output bio-openscholar/chunks.parquet \
  --method from_meta
```

Build a parquet so you can fetch chunk text quickly when serving queries.

```bash
python bio-openscholar/search_then_rerank.py \
  --index_path bio-openscholar/index_1/index.faiss \
  --meta_path bio-openscholar/index_1/meta.jsonl \
  --query "cytokine storm mitigation strategies"
```

Ensure the parquet from the previous step is accessible to any helper that reconstructs chunk text (for example, `search_with_retriever.get_chunks_by_hash`). With the index, meta, and parquet in place, run the script above to sanity-check retrieval quality.

If you want you can host this on a cloud gpu server like runpod or lambda.

## Citation

If you build on this toolkit, please cite OpenScholar:

```bibtex
@article{openscholar,
  title={{OpenScholar}: Synthesizing Scientific Literature with Retrieval-Augmented Language Models},
  author={Asai, Akari and He*, Jacqueline and Shao*, Rulin and Shi, Weijia and Singh, Amanpreet and Chang, Joseph Chee  and Lo,  Kyle and Soldaini, Luca and Feldman, Tian, Sergey and Mike, D’arcy and Wadden, David and Latzke, Matt and Minyang and Ji, Pan and Liu, Shengyan and Tong, Hao and Wu, Bohao and Xiong, Yanyu and Zettlemoyer, Luke and Weld, Dan and Neubig, Graham and Downey, Doug and Yih, Wen-tau and Koh, Pang Wei and Hajishirzi, Hannaneh},
  journal={Arxiv},
  year={2024},
}
```
