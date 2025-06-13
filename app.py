from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
from langchain_nomic import NomicEmbeddings
import pinecone
from PIL import Image
import requests
import os
from collections import defaultdict

app = FastAPI()

# OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

@app.post("/api/", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    # Step 1: Embed the question

    # Step 2: Embed the image if provided
    image_embedding = None
    if req.image:
        try:
            image_bytes = base64.b64decode(req.image)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            #embed image
        except Exception as e:
            print(f"Image embedding failed: {e}")

    # Step 3: Dense retrieval (ChromaDB)
    #get the query together nicely, maybe use gemini to describe the image

    #query the pinecone db after connecting to it before this function

    
    prompt_context = []
    for doc_id, entry in results:
        url = entry["url"]
        prompt_context.append(f"[Source: {url}]\n{entry['content'].strip()}")

    context_str = "\n---\n".join(prompt_context)
    final_prompt = [
        {"role": "system", "content": "You're a helpful TDS course assistant."},
        {"role": "user", "content": f"Context:\n\n{prompt_context}\n\nQuestion: {req.question}\n\nAnswer:"}
    ]

    response = requests.post(
        "https://aipipe.org/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": final_prompt
        }
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()

    links = []
    for doc_id, entry in results:
        url = entry["url"]
        links.append(Link(url=url, text=entry["content"].strip()[:150]))

    return QueryResponse(answer=answer, links=links)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
