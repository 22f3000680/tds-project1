import sqlite3
import json
from tqdm import tqdm
import requests
import os

"""
TODO:
- Change the model to either Jina or Nomic Atlas
- Include the embedding for image also in cleaned-discourse.json
- Add a column for image embedding in the SQLite database
"""

# Input files
INPUT_FILES = ["website-chunks.json", "cleaned-discourse.json"]
DB_FILE = "embeddings.db"

# Setup SQLite
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    source TEXT,
    url TEXT,
    content TEXT,
    embedding BLOB
)
''')
conn.commit()

# Function to get embedding
def get_embedding(text, model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")):
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": model
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error if the request failed
    data = response.json()
    return data["data"][0]["embedding"]

# Process each file
for file in INPUT_FILES:
    with open(file) as f:
        records = json.load(f)

    for rec in tqdm(records, desc=f"Embedding {file}"):
        text = rec.get("content")
        if not text:
            continue
        try:
            embedding = get_embedding(text)
            c.execute(
                "INSERT OR REPLACE INTO embeddings (id, source, url, content, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    rec["id"],
                    rec.get("filename") or rec.get("topic_slug", ""),
                    rec.get("post_url", ""),
                    text,
                    json.dumps(embedding)
                )
            )
        except Exception as e:
            print(f"Error with {rec['id']}: {e}")

    conn.commit()

conn.close()
print("Embeddings stored in", DB_FILE)
