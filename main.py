from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import base64
import io
from collections import defaultdict
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pinecone
from google import genai
from google.genai import types
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from rank_bm25 import BM25Okapi
nltk.download('punkt_tab')
nltk.download('punkt')

with open("cleaned-discourse.json") as f:
    posts = json.load(f)

# Group posts by topic_id
grouped_posts = defaultdict(list)
for post in posts:
    grouped_posts[post["topic_id"]].append(post)

# Convert to langchain documents
documents = []
for topic_id, topic_posts in grouped_posts.items():
    topic_slug = topic_posts[0]["topic_slug"]
    source_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}/"
    content = "\n\n".join(post["content"] for post in topic_posts)
    metadata = {
        "topic_id": topic_id,
        "topic_slug": topic_slug,
        "source_url": source_url
    }
    doc = Document(page_content=content, metadata=metadata)
    documents.append(doc)

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

with open('cleaned-content.json') as f:
    webcont = json.load(f)

webdocus = []
for page in webcont:
    content = page["content"]
    metadata = {
        "source_url": "https://tds.s-anand.net/#/" + page["filename"][:-3],
        "topic_slug": page["filename"],
        "topic_id": page["id"]
    }
    doc = Document(page_content=content, metadata=metadata)
    webdocus.append(doc)

docs1 = text_splitter.split_documents(webdocus)
docs.extend(docs1)

bm25_retriever = BM25Retriever.from_documents(
    docs,
    preprocess_func=word_tokenize,
    k=5
)

load_dotenv()

app = FastAPI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Setup Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY") 
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = "tds-virtual-ta" 
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"), openai_api_base=os.getenv("OPENAI_BASE_URL"))
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

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
    # Get image description if image is provided
    query = req.question
    image = req.image
    if image:
        client = genai.Client()
        prompt = "Extract the text from this image."
        image_data = base64.b64decode(image)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
            ]
        )
        query += f" {response.candidates[0].content.text}"

    # Vector search
    # Vector search with scores
    vector_docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    vector_docs = [doc for doc, score in vector_docs_with_scores]
    vector_scores = [score for doc, score in vector_docs_with_scores]

    # BM25 search (with scores)
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:5]
    bm25_docs = [docs[i] for i in bm25_indices]
    bm25_scores_list = [bm25_scores[i] for i in bm25_indices]

    # Normalize scores
    scaler = MinMaxScaler()
    vector_scores_norm = scaler.fit_transform(np.array(vector_scores).reshape(-1, 1)).flatten()
    bm25_scores_norm = scaler.fit_transform(np.array(bm25_scores_list).reshape(-1, 1)).flatten()

    # Combine scores
    combined_scores = {}
    for i, doc in enumerate(vector_docs):
        url = doc.metadata.get('source_url', '')
        combined_scores[url] = combined_scores.get(url, 0) + 0.5 * vector_scores_norm[i]
    for i, doc in enumerate(bm25_docs):
        url = doc.metadata.get('source_url', '')
        combined_scores[url] = combined_scores.get(url, 0) + 0.5 * bm25_scores_norm[i]

    # Sort and deduplicate
    all_docs = vector_docs + bm25_docs
    all_docs.sort(key=lambda doc: combined_scores.get(doc.metadata.get('source_url', ''), 0), reverse=True)
    seen_urls = set()
    top_docs = []
    for doc in all_docs:
        url = doc.metadata.get('source_url', '')
        if url not in seen_urls:
            top_docs.append(doc)
            seen_urls.add(url)
        if len(top_docs) >= 5:
            break


    # Format documents for prompt
    context = "\n---\n".join(
        f"[Source: {doc.metadata.get('source_url', '')}]\n{doc.page_content.strip()}" for doc in top_docs
    )
    print(context)

    # Prompt and LLM chain
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant for the Tools in Data Science course.
        Answer the following question using the context below.

        Context:
        {context}

        Question: {question}
        Return a JSON object with "answer" from the context.
        """
    )
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    chain = prompt | llm | JsonOutputParser(keys=["answer"])
    result = chain.invoke({"context": context, "question": query})

    # Build response
    return QueryResponse(
        answer=result['answer'],
        links=[Link(url=doc.metadata.get('source_url', ''), text=doc.page_content.strip()[:150]) for doc in top_docs]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
