# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "nomic",
#   "chromadb",
#   "tqdm",
#   "requests",
# ]
# ///

import os
import json
import requests
import uuid
import base64
from nomic import atlas, embed
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# --- Load data ---
with open("cleaned-discourse.json") as f:
    discourse_data = json.load(f)

with open("website-chunks.json") as f:
    markdown_data = json.load(f)

# --- Setup ChromaDB ---
client = chromadb.PersistentClient(path="./chroma_store", settings=Settings())
collection = client.get_or_create_collection("tds_multimodal")

# --- Helper Functions ---
def embed_text(text):
    output = embed.text(
        texts=[text],
        model='nomic-embed-text-v1.5',
        task_type='search_document'
    )
    return output['embeddings'][0]

def embed_image_from_base64(b64str):
    try:
        image_bytes = base64.b64decode(b64str)
        output = embed.image(
            images=[image_bytes],
            model='nomic-embed-vision-v1.5'
        )
        return output['embeddings'][0]
    except Exception as e:
        print(f"Failed to embed base64 image: {e}")
        return None

# --- Process Discourse Posts ---
for post in tqdm(discourse_data, desc="Discourse posts"):
    post_id = str(post["id"])
    source_url = "https://discourse.onlinedegree.iitm.ac.in" + post["post_url"]
    text = post.get("content", "")
    images_b64 = post.get("image", [])

    # Embed text
    if text:
        text_embedding = embed_text(text)
        collection.add(
            ids=[f"{post_id}-text"],
            documents=[text],
            embeddings=[text_embedding],
            metadatas=[{
                "post_id": post_id,
                "source_url": source_url,
                "type": "text"
            }]
        )

    # Embed images from base64
    for i, img_b64 in enumerate(images_b64):
        emb = embed_image_from_base64(img_b64)
        if emb:
            collection.add(
                ids=[f"{post_id}-img{i}"],
                documents=[""],
                embeddings=[emb],
                metadatas=[{
                    "post_id": post_id,
                    "source_url": source_url,
                    "type": "image",
                    "position": i
                }]
            )

# --- Process Markdown Files ---
for doc in tqdm(markdown_data, desc="Markdown chunks"):
    doc_id = doc["id"]
    content = doc["content"]
    filename = doc.get("filename", "")
    text_embedding = embed_text(content)

    collection.add(
        ids=[doc_id],
        documents=[content],
        embeddings=[text_embedding],
        metadatas=[{
            "post_id": doc_id,
            "source_url": filename,
            "type": "text",
            "filename": filename
        }]
    )

print("Done storing text and image embeddings into ChromaDB.")
