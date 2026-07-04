# RAG Pipeline Practice

This repository contains my practice implementation of a basic Retrieval-Augmented Generation (RAG) pipeline while learning AI Engineering.

The goal is to understand how each component of a RAG system works, including document loading, chunking, embedding generation, vector storage, and semantic retrieval using local models.

---

## Tech Stack

- Python
- LangChain
- Ollama
- ChromaDB
- UV
- python-dotenv

---

## Folder Structure

```
RAG_PIPELINE
│
├── docs/
│   ├── Google.txt
│   ├── Microsoft.txt
│   └── SpaceX.txt
│
├── src/
│   ├── chroma_db/
│   ├── ingestion_pipeline.py
│   ├── retrieval_pipeline.py
│   └── test+ollama.py
│
├── .env
├── pyproject.toml
├── uv.lock
├── README.md
```

---

## Learning Objectives

- Load documents from a directory
- Split documents into chunks
- Generate embeddings using Ollama (`nomic-embed-text`)
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Understand the complete RAG ingestion and retrieval workflow

---

## RAG Workflow

```
Documents
    │
    ▼
Directory Loader
    │
    ▼
Text Splitter
    │
    ▼
Embeddings (Ollama)
    │
    ▼
ChromaDB
    │
    ▼
Retriever
    │
    ▼
Relevant Documents
```

---

## Setup

### Install dependencies

```bash
uv sync
```

### Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### Verify installation

```bash
ollama list
```

---

## Run the Ingestion Pipeline

```bash
python src/ingestion_pipeline.py
```

This script:

- Loads text documents
- Splits them into chunks
- Generates embeddings
- Stores them in ChromaDB

---

## Run the Retrieval Pipeline

```bash
python src/retrieval_pipeline.py
```

Enter a query, and the pipeline retrieves the most relevant document chunks using semantic search.

---

## Current Status

- ✅ Document Loading
- ✅ Chunking
- ✅ Embedding Generation
- ✅ ChromaDB Storage
- ✅ Semantic Retrieval

---

## Next Steps

- Response generation using an LLM
- Prompt templates
- Complete end-to-end RAG chain
- Conversation memory
- Hybrid search
- Reranking

---

## Notes

This repository is intended for learning and experimenting with RAG concepts. It is not a production-ready application.