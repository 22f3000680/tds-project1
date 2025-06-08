# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "chromadb",
#   "whoosh",
# ]
# ///

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import os
import chromadb
from chromadb.config import Settings

# Define schema for Whoosh
schema = Schema(
    id=ID(stored=True, unique=True),
    content=TEXT(stored=True),
    url=ID(stored=True)
)

# Prepare index directory
index_dir = "whoosh_index"
if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    ix = create_in(index_dir, schema)
else:
    from whoosh.index import open_dir
    ix = open_dir(index_dir)

# Load cleaned Chroma collection
client = chromadb.PersistentClient(path="./chroma_store", settings=Settings())
collection = client.get_collection("tds_multimodal")
data = collection.get(include=["metadatas", "documents"])

# Index documents with metadata
writer = ix.writer()
for doc_id, content, meta in zip(data["ids"], data["documents"], data["metadatas"]):
    if not content or not content.strip():
        continue
    url = meta.get("source_url", "")
    writer.add_document(id=doc_id, content=content, url=url)
writer.commit()

print(f"Indexed {len(data['ids'])} documents into Whoosh BM25 index")
