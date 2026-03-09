import csv
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
    words = text.split()
    chunks = []

    start = 0
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


def load_txt_files(folder: str) -> list[tuple[str, str]]:
    files = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        files.append((path.name, text))
    return files


def main():
    input_folder = "data/raw/shakespeare"
    output_chunks = "data/processed/chunks.json"
    output_embeddings = "data/processed/embeddings.csv"

    chunk_size = 500
    overlap = 75

    # ensure processed directory exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    docs = load_txt_files(input_folder)

    if not docs:
        raise ValueError(f"No .txt files found in {input_folder}")

    all_chunks = []
    next_id = 0

    for filename, text in docs:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            all_chunks.append({
                "id": next_id,
                "source": filename,
                "text": chunk
            })
            next_id += 1

    print(f"Loaded {len(docs)} files")
    print(f"Created {len(all_chunks)} chunks")

    texts = [entry["text"] for entry in all_chunks]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    with open(output_chunks, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    with open(output_embeddings, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, emb in enumerate(embeddings):
            writer.writerow([i] + emb.tolist())

    print(f"Saved chunks to {output_chunks}")
    print(f"Saved embeddings to {output_embeddings}")
    print(f"Embedding dimension: {len(embeddings[0])}")


if __name__ == "__main__":
    main()