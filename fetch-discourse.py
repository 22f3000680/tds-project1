import requests
import time
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
cookie = os.getenv("DISCOURSE_COOKIE")

BASE = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY = "c/courses/tds-kb/34"

COOKIES = {"_t": cookie}
HEADERS = {
    "User-Agent": "TDS scraper",
    "Referer": BASE,
}
DATE_FORMAT = "%Y-%m-%d"

def fetch_all_posts(start_str, end_str):
    start_date = datetime.strptime(start_str, DATE_FORMAT).date()
    end_date = datetime.strptime(end_str, DATE_FORMAT).date()

    posts = []
    page = 0
    break_outer = False

    while not break_outer:
        url = f"{BASE}/{CATEGORY}.json?page={page}"
        print(f"Fetching page {page}...")

        res = requests.get(url, headers=HEADERS, cookies=COOKIES)
        if res.status_code != 200:
            print(f"Error {res.status_code} on page {page}")
            break

        data = res.json()
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            print("No more topics.")
            break

        for topic in topics:
            topic_id = topic["id"]
            last_posted = topic.get("last_posted_at", topic.get("created_at", ""))[:10]
            topic_last_date = datetime.strptime(last_posted, DATE_FORMAT).date()

            if topic_last_date < start_date:
                print(f"Skipping old topic {topic_id} (last posted: {last_posted})")
                break_outer = True
                break

            # Fetch full topic details
            topic_url = f"{BASE}/t/{topic_id}.json"
            tdata = requests.get(topic_url, headers=HEADERS, cookies=COOKIES).json()

            for post in tdata.get("post_stream", {}).get("posts", []):
                post_date = datetime.strptime(post["created_at"][:10], DATE_FORMAT).date()
                if start_date <= post_date <= end_date:
                    posts.append(post)

        page += 1
        time.sleep(0.5)

    print(f"Total posts collected: {len(posts)}")
    return posts


raw_posts = fetch_all_posts("2025-01-01", "2025-04-15")


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


def remove_html_tags(content):
    """Remove HTML tags from content, returning plain text."""
    return BeautifulSoup(content, 'html.parser').get_text()

def process_chunk(chunk):
    """Process a single JSON chunk: extract image URLs, encode images, add 'image' key."""
    content = chunk.get("content", "")
    image_urls = extract_image_urls(content)
    chunk["image"] = image_urls
    chunk["content"] = remove_html_tags(content)
    return chunk

processed_posts = []
for post in good_posts:
    processed_posts.append(process_chunk(post))

# Save to new file
with open("cleaned-discourse.json", "w") as f:
    json.dump(processed_posts, f, indent=2)

print(f"Converted {len(processed_posts)} posts.")
