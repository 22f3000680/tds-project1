# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "chromadb",
#   ]
# ///

import chromadb
from chromadb.config import Settings
import json

# Setup ChromaDB client
client = chromadb.PersistentClient(path="./chroma_store", settings=Settings())

# Load existing collection
old_collection = client.get_collection("tds_multimodal")
all_data = old_collection.get(include=["metadatas", "documents", "embeddings"])

# Filter data
filtered_data = {
    "ids": [],
    "documents": [],
    "embeddings": [],
    "metadatas": []
}

for _id, doc, meta, emb in zip(
    all_data["ids"], all_data["documents"], all_data["metadatas"], all_data["embeddings"]):

    if not doc or not doc.strip():
        continue

    if not meta.get("post_id") or not meta.get("source_url"):
        continue

    filtered_data["ids"].append(_id)
    filtered_data["documents"].append(doc)
    filtered_data["embeddings"].append(emb)
    filtered_data["metadatas"].append(meta)

# Delete and recreate the collection
client.delete_collection("tds_multimodal")
new_collection = client.create_collection("tds_multimodal")

# Insert cleaned entries
batch_size = 100
for i in range(0, len(filtered_data["ids"]), batch_size):
    new_collection.add(
        ids=filtered_data["ids"][i:i+batch_size],
        documents=filtered_data["documents"][i:i+batch_size],
        embeddings=filtered_data["embeddings"][i:i+batch_size],
        metadatas=filtered_data["metadatas"][i:i+batch_size]
    )

print(f"Cleaned and reloaded {len(filtered_data['ids'])} entries into tds_multimodal")
