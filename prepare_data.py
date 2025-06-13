from collections import defaultdict
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm 
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_nomic import NomicEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone


with open("cleaned-discourse.json") as f:
    posts = json.load(f)

# Group posts by topic_id
grouped_posts = defaultdict(list)
for post in posts:
    grouped_posts[post["topic_id"]].append(post)

#convert to langchain documents
documents = []
for topic_id, topic_posts in grouped_posts.items():
    # Get the topic_slug from the first post (all should be the same for a topic_id)
    topic_slug = topic_posts[0]["topic_slug"]
    # Build the source_url
    source_url = f"https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34/t/{topic_slug}/{topic_id}/"
    # Concatenate all post contents
    content = "\n\n".join(post["content"] for post in topic_posts)
    # Create metadata
    metadata = {
        "topic_id": topic_id,
        "topic_slug": topic_slug,
        "source_url": source_url
    }
    # Create the document
    doc = Document(page_content=content, metadata=metadata)
    documents.append(doc)

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Adjust based on your needs
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
        "topic_id" : page["id"]
    }
    doc = Document(page_content=content, metadata=metadata)
    webdocus.append(doc)

docs1 = text_splitter.split_documents(webdocus)
docs.extend(docs1)

#setting up environment variables
load_dotenv()

#embedding and storing in Pinecone
embeddings = NomicEmbeddings(model = "nomic-embed-text-v1.5")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=pinecone_api_key)

index_name = "tds-virtual-ta"
dimension = 768  

index = pc.Index(index_name) 

batch_size = 100 

for i in tqdm(range(0, len(docs), batch_size)):
    batch = docs[i:i+batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        embedding=embeddings,
        index_name=index_name
    )