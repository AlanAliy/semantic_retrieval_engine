from sentence_transformers import SentenceTransformer
import csv
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print('usage: python embed_query.py "your query here"')
    sys.exit(1)

query = sys.argv[1]

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(query)

output_path = Path("data/processed/query.csv")

# ensure directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(embedding.tolist())

print(f"Saved query embedding to {output_path}")