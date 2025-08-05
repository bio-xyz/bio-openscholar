# OpenScholar: A Scientific RAG System with Llama 3.1

## 🚀 Project Overview

OpenScholar is an advanced Retrieval-Augmented Generation (RAG) system designed to answer complex questions based on a custom library of scientific documents. This project implements a sophisticated, multi-stage pipeline to ensure that the answers are not only accurate and relevant but also properly cited.

The system is built using a state-of-the-art stack, including a fine-tuned Contriever model for initial document retrieval, a BGE-Reranker to refine the search results, and a fine-tuned Llama 3.1 8B model to generate human-like answers.

### Key Features

- **End-to-End Pipeline**: From raw text documents to a fully functional question-answering system.
- **Fine-Tuned Components**: Each part of the RAG pipeline (retriever, reranker, and generator) is fine-tuned on your custom data for optimal performance.
- **High-Quality Answers**: Leverages the power of Llama 3.1 for nuanced and context-aware responses.
- **Efficient and Scalable**: Uses FAISS for fast similarity search and 4-bit quantization for efficient model loading.

## 🛠️ Technologies Used

- **Core Frameworks**: PyTorch, Hugging Face Transformers, Datasets, PEFT (Parameter-Efficient Fine-Tuning)
- **Language Models**:
  - **Generator**: meta-llama/Llama-3.1-8B-Instruct
  - **Retriever**: facebook/contriever
  - **Reranker**: BAAI/bge-reranker-large
  - **QA Data Generation**: google/flan-t5-large
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Experiment Tracking**: Weights & Biases (wandb)

## ⚙️ How It Works: The RAG Pipeline

The project is structured as a series of sequential steps, each building upon the last to create the final RAG system.

### 1. Data Preparation

- **Loading**: The system starts by loading your scientific documents from a specified directory (e.g., .md files).
- **Chunking**: Each document is split into smaller, 250-word passages. This is a crucial step to ensure that the retriever can find the most relevant snippets of information.

### 2. Retriever Fine-Tuning (Contriever)

- **Goal**: To teach a retriever model to understand the relationships between passages in your documents.
- **Method**: The facebook/contriever model is fine-tuned using a self-supervised approach. It learns that passages that are close to each other in a document are semantically related.

### 3. Indexing for Search (FAISS)

- **Goal**: To create a fast and efficient search index of all your document passages.
- **Method**: The fine-tuned retriever model is used to generate a vector embedding for every passage. These embeddings are then stored in a FAISS index, which allows for lightning-fast similarity searches.

### 4. Reranker Fine-Tuning (BGE-Reranker)

- **Goal**: To improve the accuracy of the search results by re-ordering the retrieved passages.
- **Method**: The BAAI/bge-reranker-large model is fine-tuned on synthetic data. It learns to distinguish between highly relevant passages and less relevant ones, pushing the best results to the top.

### 5. Synthetic QA Data Generation

- **Goal**: To create a high-quality dataset for fine-tuning the final generator model.
- **Method**: The google/flan-t5-large model is used to automatically generate question-answer pairs based on the content of your documents. This step is critical for teaching the Llama model how to answer questions in the style you want.

### 6. Generator Fine-Tuning (Llama 3.1)

- **Goal**: To teach the Llama 3.1 model to act as the "brain" of the RAG system.
- **Method**: The meta-llama/Llama-3.1-8B-Instruct model is fine-tuned using the synthetic QA dataset. It learns to synthesize a final answer from the context provided by the retriever and reranker, complete with citations. QLoRA is used for memory-efficient fine-tuning.

### 7. Final Inference Pipeline

**Putting It All Together**: The final step combines all the fine-tuned components into a single, cohesive system. When you ask a question:

- **The Retriever** finds the top N relevant passages from the FAISS index.
- **The Reranker** re-orders these passages to prioritize the most relevant ones.
- **The Generator** (Llama 3.1) receives the question and the top-ranked passages and generates a final, cited answer.

## 📝 Important Notes

> [!NOTE]
>
> - **GPU Requirements**: This project requires a powerful GPU, especially for fine-tuning the Llama 3.1 model. An NVIDIA A100 or H100 is recommended.
> - **Data Quality**: The quality of the final answers is highly dependent on the quality of the documents you provide and the synthetic QA data generated. If you get poor results, consider improving the QA generation step.
> - **Dependency Conflicts**: The notebook may have a conflict between fastai and torch. If you encounter issues, it's best to start with a clean environment and install only the packages listed in the first cell.
