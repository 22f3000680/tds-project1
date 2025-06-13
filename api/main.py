from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import base64
import io
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pinecone
from google import genai
from google.genai import types

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
    #get image description if image is provided
    query = req.question
    image = req.image
    if image:
        client = genai.Client()
        prompt = "Extract the text from this image."
        image_data = base64.b64decode(image)
        # Call the model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
            ]
        )
        query += f" {response.candidates[0].content.text}"


    # Vector search
    docs: List[Document] = vectorstore.similarity_search(query, k=2)

    # Format documents for prompt
    context = "\n---\n".join(
        f"[Source: {doc.metadata.get('source_url', '')}]\n{doc.page_content.strip()}" for doc in docs[:2]
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant for the Tools in Data Science course.
        Answer the following question using the context below. If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question: {question}
        Return a JSON object with "answer" and "links" (list of {{"url": ..., "text": ...}}) from the context.
        """
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    chain = prompt | llm | JsonOutputParser(
        keys=["answer", "links"],
        link_model=Link
    )
    result = chain.invoke({"context": context, "question": query})

    # Build response
    return QueryResponse(answer=result['answer'], links=result['links'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
