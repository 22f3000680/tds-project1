# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "bs4",
# ]
# ///

import re
from bs4 import BeautifulSoup
import json

def clean_text_pipeline(text):
    """
    Pipeline to clean text by:
    1. Removing HTML comments and tags
    2. Removing YouTube links
    3. Normalizing whitespace and special characters
    """
    # 1. Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # 2. Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')
    # 3. Remove YouTube links
    youtube_pattern = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+'
    text = re.sub(youtube_pattern, '', text)
    # 4. Normalize whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(ch for ch in text if ch.isprintable())
    return text.strip()

with open("website-chunks.json", "r") as file:
    website_chunks = json.load(file)

for chunk in website_chunks:
    if 'content' in chunk:
        original_content = chunk['content']
        cleaned_content = clean_text_pipeline(original_content)
        chunk['content'] = cleaned_content

with open("cleaned-content.json", "w") as file:
    json.dump(website_chunks, file, indent=2)