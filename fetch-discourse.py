import requests
import time
import os
from datetime import datetime

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


posts = fetch_all_posts("2025-01-01", "2025-04-15")
import json
with open("discourse-posts.json", "w") as f:
    json.dump(posts, f, indent=2)