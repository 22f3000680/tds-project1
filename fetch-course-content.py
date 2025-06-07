from pathlib import Path
import uuid
import json
import tiktoken
import re

# Setup
repo_dir = Path("/home/krithika/tools-in-data-science-public/2025-01")
sidebar_file = repo_dir / "_sidebar.md"
output_file = "website-chunks.json"

# Tokenizer
encoding = tiktoken.encoding_for_model("text-embedding-3-small")
MAX_TOKENS = 8192
SAFE_TOKENS = 3000  # safer chunk size

# Step 1: Get markdown files from sidebar
markdown_files = []
with open(sidebar_file) as f:
    for line in f:
        if "(" in line and line.strip().endswith(".md)"):
            path = line.split("(")[-1].rstrip(")\n")
            markdown_files.append(repo_dir / path)

print(f"Found {len(markdown_files)} markdown files.")

# Step 2: Split large markdowns into safe chunks
def split_markdown_by_token(content, filename):
    chunks = []
    buffer = ""
    for block in re.split(r"(?=\n## |\n<details>)", content):
        candidate = buffer + block
        if len(encoding.encode(candidate)) < SAFE_TOKENS:
            buffer = candidate
        else:
            if buffer:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "filename": filename,
                    "content": buffer.strip()
                })
            buffer = block
    if buffer:
        chunks.append({
            "id": str(uuid.uuid4()),
            "filename": filename,
            "content": buffer.strip()
        })
    return chunks

# Step 3: Process all files and save
all_chunks = []
for file in markdown_files:
    with open(file, encoding="utf-8") as f:
        content = f.read()
        all_chunks.extend(split_markdown_by_token(content, file.name))

with open(output_file, "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Done. Saved {len(all_chunks)} chunks to {output_file}")
