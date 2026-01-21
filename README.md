# Local RAG from Scratch

A fully local Retrieval-Augmented Generation (RAG) system built step by step using Python, FAISS, and Ollama.

This project demonstrates how RAG actually works internally, without relying on unstable or opaque LangChain chains.

---

## 🔍 What this project demonstrates

- End-to-end RAG pipeline (PDF → embeddings → retrieval → answer)
- Fully local LLM using Ollama (no cloud APIs)
- FAISS vector search with persistent index
- Hallucination-resistant answers ("I don't know" when context is missing)
- Modular, production-style Python codebase
- Real-world debugging decisions (chunking, `k` tuning, retriever APIs)

---

## 🧠 Architecture Overview

1. Load PDF documents
2. Split text into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve top-k relevant chunks
6. Inject context into LLM prompt
7. Uses a local LLM (Ollama) to answer questions
8. Generate grounded answer: Avoids hallucination by answering **only from context**

---

## Tech Stack

- Python 3.10
- FAISS (vector search)
- Sentence Transformers (embeddings)
- Ollama (local LLM)
- LangChain (minimal, explicit usage)

---

## 📂 Project Structure

    ├── main.py            # Entry point (ask questions)
    ├── rag_pipeline.py    # Retrieval + prompting logic
    ├── loaders.py         # PDF loading
    ├── chunking.py        # Text splitting
    ├── embeddings.py      # Embeddings + FAISS
    ├── llm.py             # Ollama LLM wrapper
    ├── requirements.txt
    ├── data/              # PDFs (ignored in git)
    └── vector_index/      # FAISS index (ignored in git)


---

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
