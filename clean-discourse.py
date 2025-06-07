# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "bs4",
#   "requests",
# ]
# ///

import json
import re
import base64
import requests
from bs4 import BeautifulSoup

# Load scraped posts from JSON
with open("discourse-posts.json") as f:
    raw_posts = json.load(f)

# Clean and convert to minimal format
def convert_post(post):
    return {
        "id": post["id"],
        "topic_id": post["topic_id"],
        "post_url": post.get("post_url", ""),
        "topic_slug": post.get("topic_slug", ""),
        "post_number": post["post_number"],
        "created_at": post["created_at"],
        "username": post.get("username", ""),
        "content": post.get("cooked", "")
    }

cleaned_posts = [convert_post(p) for p in raw_posts]
good_posts = []

for post in cleaned_posts:
    if post["content"] == "":
        continue
    # Clean up content by removing HTML tags and unnecessary whitespace
    post["content"] = post["content"].strip()
    post["content"] = ' '.join(post["content"].split())
    good_posts.append(post)


def extract_image_urls(content):
    """Extract all image URLs from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    img_urls = [img['src'] for img in soup.find_all('img') if img.has_attr('src')]
    a_urls = [a['href'] for a in soup.find_all('a') if a.has_attr('href') and a['href'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
    return list(set(img_urls + a_urls))

def encode_image_to_base64(url):
    """Download image and encode in base64."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"Failed to encode image {url}: {e}")
        return None

def remove_html_tags(content):
    """Remove HTML tags from content, returning plain text."""
    return BeautifulSoup(content, 'html.parser').get_text()

def process_chunk(chunk):
    """Process a single JSON chunk: extract image URLs, encode images, add 'image' key."""
    content = chunk.get("content", "")
    image_urls = extract_image_urls(content)
    encoded_images = [encode_image_to_base64(url) for url in image_urls]
    chunk["image"] = [img for img in encoded_images if img is not None]
    chunk["content"] = remove_html_tags(content)
    return chunk

processed_posts = []
for post in good_posts:
    processed_posts.append(process_chunk(post))

# Save to new file
with open("cleaned-discourse.json", "w") as f:
    json.dump(processed_posts, f, indent=2)

print(f"Converted {len(processed_posts)} posts.")
