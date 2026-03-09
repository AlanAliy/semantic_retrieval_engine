# Semantic Document Retrieval Engine

A **high-performance semantic search engine** that retrieves relevant text passages using **transformer embeddings and vector similarity search**.

The system combines **Python-based embedding generation** with a **parallelized C++ search engine** to enable efficient retrieval across large document collections.

---

# Overview

Traditional search engines rely on keyword matching. This project implements **semantic search**, where documents are retrieved based on **meaning rather than exact words**.

The system works by:

1. Splitting documents into smaller chunks  
2. Generating vector embeddings for each chunk using a transformer model  
3. Indexing those embeddings  
4. Searching for the closest vectors to a query embedding  

This architecture mirrors the **retrieval component of Retrieval-Augmented Generation (RAG)** systems used in modern AI applications.

---

# Architecture

```
Documents (.txt)
        │
        ▼
Chunking Pipeline (Python)
        │
        ▼
Embedding Generation (Sentence Transformers)
        │
        ▼
Embedding Storage (CSV)
        │
        ▼
C++ Vector Index
        │
        ▼
Parallel Similarity Search (OpenMP)
        │
        ▼
Top-k Relevant Chunks
```

---

# Features

- Semantic document search using **transformer embeddings**
- High-performance **C++ vector similarity search engine**
- Support for multiple distance metrics:
  - Cosine similarity
  - L2 distance
  - Manhattan distance
- **Parallel query execution using OpenMP**
- Document ingestion pipeline with configurable chunking
- Hybrid **Python + C++ architecture**

---

# Technologies Used

### Languages
- C++
- Python

### Libraries
- sentence-transformers
- OpenMP

### Concepts
- Retrieval-Augmented Generation (RAG)
- Vector similarity search
- Embedding-based retrieval
- Parallel computing

---

# Project Structure

```
vector_search/
│
├── src/
│   ├── main.cpp
│   ├── vectorIndex.cpp
│   ├── chunkStore.cpp
│   ├── embeddingLoader.cpp
|
├── include/
│   ├── vectorIndex.h
│   ├── chunkStore.h
│   ├── embeddingLoader.h
|
├── scripts/
│   ├── build_embeddings.py
│   ├── semantic_search.py
│
├── data/
│   ├── documents/
│   ├── chunks.json
│   ├── embeddings.csv
│
├── Makefile
└── README.md
```

---

# How It Works

## 1. Document Chunking

Documents are split into overlapping chunks so embeddings represent manageable pieces of text.

Example:

```
Chunk 1: words 1–100
Chunk 2: words 80–180
Chunk 3: words 160–260
```

Overlapping chunks help preserve semantic context.

---

## 2. Embedding Generation

Chunks are converted into dense vector embeddings using the transformer model:

```
all-MiniLM-L6-v2
```

Each chunk becomes a **384-dimensional vector** representing its semantic meaning.

---

## 3. Vector Index

Embeddings are stored in a **contiguous memory vector index** implemented in C++.

This allows fast similarity computation across large datasets.

---

## 4. Query Processing

A query is processed by:

1. Generating an embedding for the query
2. Computing similarity against all indexed vectors
3. Returning the **top-k most similar chunks**

Similarity is computed using:

```
cosine similarity
L2 distance
Manhattan distance
```

---

## 5. Parallel Search

Similarity search is parallelized across CPU cores using **OpenMP**:

```cpp
#pragma omp parallel for
for each vector in index
    compute distance
```

This significantly improves query performance on large datasets.

---

# Running the Project

## 1. Install Python Dependencies

```bash
pip install sentence-transformers
```

---

## 2. Build the C++ Engine

```bash
make
```

This compiles the semantic search engine.

---

## 3. Generate Embeddings

Place `.txt` files in the `data/` directory and run:

```bash
python scripts/build_embeddings.py
```

This produces:

```
embeddings.csv
chunks.json
```

---

## 4. Run a Query

```bash
python scripts/semantic_search.py "love and despair"
```

Example output:

```
ID: 502
Score: 0.297763
Source: t8.shakespeare.txt
Text: a horse. Prince. Hark how hard he fetches breath...
```

---

# Example Use Cases

- Semantic document search
- Retrieval for LLM pipelines
- Research over text corpora
- Knowledge base search
- Retrieval-Augmented Generation (RAG) systems

---

# Future Improvements

Potential improvements include:

- SIMD vectorization for faster distance computation
- Approximate nearest neighbor indexing
- REST API interface for real-time search
- GPU acceleration
- Integration with LLM pipelines

---

# Author

**Alan Ali Yusuf**  
University of Southern California  
Computer Engineering & Computer Science
