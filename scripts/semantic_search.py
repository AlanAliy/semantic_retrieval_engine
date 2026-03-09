import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from sentence_transformers import SentenceTransformer


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/search.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]

    root = Path(__file__).resolve().parent.parent
    embeddings_file = root / "data/processed/embeddings.csv"
    chunks_file = root / "data/processed/chunks.json"
    binary = root / "build/semantic_search"

    if not embeddings_file.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_file}")

    if not chunks_file.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_file}")

    if not binary.exists():
        raise FileNotFoundError(f"Missing C++ binary: {binary}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(query)

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        suffix=".csv",
        delete=False,
        encoding="utf-8"
    ) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(embedding.tolist())
        query_file = tmp.name

    try:
        result = subprocess.run(
            [str(binary), str(embeddings_file), str(chunks_file), query_file],
            check=True,
            text=True,
            capture_output=True
        )

        print(result.stdout)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

    finally:
        Path(query_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()