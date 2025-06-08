# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "nomic",
#   "chromadb",
#   "fastapi",
#   "pydantic",
#   "pillow",
#   "uvicorn",
#   "requests",
# ]
# ///

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import base64
import io
import chromadb
from nomic import embed
from PIL import Image
import requests
import os
from collections import defaultdict

app = FastAPI()

# ChromaDB setup
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection("tds_multimodal")

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class QueryRequest(BaseModel):
    question: str
    image: str = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

@app.post("/api/", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    # Step 1: Embed the question
    question_embedding = embed.text(
        texts=[req.question],
        model="nomic-embed-text-v1.5",
        task_type="search_query"
    )["embeddings"][0]

    # Step 2: Embed the image if provided
    image_embedding = None
    if req.image:
        try:
            image_bytes = base64.b64decode(req.image)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            image_embedding = embed.image(
                images=[buffer.read()],
                model="nomic-embed-vision-v1.5"
            )["embeddings"][0]
        except Exception as e:
            print(f"Image embedding failed: {e}")

    # Step 3: Query ChromaDB
    query_embeddings = [question_embedding]
    if image_embedding:
        query_embeddings.append(image_embedding)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=10
    )

    # Step 4: Merge contexts by post_id
    context_by_post = defaultdict(lambda: {"content": [], "url": ""})
    for docs, metas in zip(results["documents"], results["metadatas"]):
        for doc, meta in zip(docs, metas):
            if not doc.strip():
                continue  # skip empty docs
            post_id = meta.get("post_id")
            if not post_id:
                continue
            context_by_post[post_id]["content"].append(doc)
            if "source_url" in meta:
                context_by_post[post_id]["url"] = meta["source_url"]

    # Sort by relevance using order in results["ids"] (same as relevance order)
    seen = set()
    sorted_contexts = []
    for ids in results["ids"]:
        for pid in ids:
            base_pid = pid.split("-")[0]
            if base_pid not in seen:
                seen.add(base_pid)
                sorted_contexts.append(base_pid)

    # Step 5: Build context and links (top 2 only)
    contexts = []
    links = []
    for post_id in sorted_contexts:
        content_data = context_by_post[post_id]
        if not content_data["content"] or not content_data["url"]:
            continue  # skip entries with no text or link

        context = "\n".join(content_data["content"])
        url = content_data["url"]
        if url.endswith(".md"):
            url = "https://tds.s-anand.net/#/" + url[:-3]
        contexts.append(context)
        links.append({
            "url": url,
            "text": context[:150]
        })
        if len(contexts) == 2:
            break

    # Step 6: Use OpenAI to answer the question using context
    context_str = "\n---\n".join(contexts)
    prompt = f"Answer the question using the following context:\n\n{context_str}\n\nQuestion: {req.question}\nAnswer:"

    response = requests.post(
        "https://aipipe.org/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()

    return QueryResponse(answer=answer, links=links)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port = 8000)