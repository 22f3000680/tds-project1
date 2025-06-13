from collections import defaultdict
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings 
from langchain_pinecone import PineconeVectorStore
import pinecone

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

# Setting up environment variables
load_dotenv()

# Embedding and storing in Pinecone
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"), openai_api_base=os.getenv("OPENAI_BASE_URL"))  

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=pinecone_api_key)

index_name = "tds-virtual-ta"
dimension = 1536  # OpenAI embedding dimension for text-embedding-ada-002

index = pc.Index(index_name)

batch_size = 100

for i in tqdm(range(0, len(docs), batch_size)):
    batch = docs[i:i+batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        embedding=embeddings,
        index_name=index_name
    )
