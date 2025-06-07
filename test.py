# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "nomic",
#   "chromadb",
#   "tqdm",
#   "requests",
#   "pillow",
# ]
# ///

import json
import base64
from nomic import embed
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

# Load discourse data with base64 images
with open("cleaned-discourse.json") as f:
    discourse_data = json.load(f)

# Setup ChromaDB
client = chromadb.PersistentClient(path="./chroma_store", settings=Settings())
collection = client.get_or_create_collection("tds_multimodal")

# Track results
success_count = 0
fail_count = 0

# Helper to check and embed base64 image
def verify_and_embed_image(img_url, post_id, img_index, source_url):
    try:
        # Attempt to embed with Nomic
        result = embed.image(
            images=[img_url],
            model='nomic-embed-vision-v1.5'
        )
        emb = result['embeddings'][0]
        print(f"✅ Post {post_id}, image {img_index}: Embedded successfully (len={len(emb)})")

        # Add to ChromaDB
        collection.add(
            ids=[f"{post_id}-img{img_index}"],
            documents=["[image]"],
            embeddings=[emb],
            metadatas=[{
                "post_id": post_id,
                "source_url": source_url,
                "type": "image",
                "position": img_index
            }]
        )
        return True

    except Exception as e:
        print(f"❌ Post {post_id}, image {img_index}: Failed")
        return False

# Process each post
for post in tqdm(discourse_data, desc="Checking images"):
    post_id = str(post["id"])
    source_url = "https://discourse.onlinedegree.iitm.ac.in" + post["post_url"]
    images = post.get("image", [])

    for i, img in enumerate(images):
        result = verify_and_embed_image(img, post_id, i, source_url)
        if result:
            success_count += 1
        else:
            fail_count += 1

print("\n✅ Total successful image embeddings:", success_count)
print("❌ Total failed image embeddings:", fail_count)
