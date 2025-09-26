import os
import re
import sys
import time
import mysql.connector
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image
import pytesseract
import praw
from wordcloud import STOPWORDS
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "phpmyadmin")
DB_PASS = os.getenv("DB_PASS", "root")
DB_NAME = os.getenv("DB_NAME", "reddit_scraper")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "reddit-scraper-lab"

def init_db():
    conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cur.execute(f"USE {DB_NAME}")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id VARCHAR(20) PRIMARY KEY,
            title TEXT,
            body TEXT,
            author_masked VARCHAR(50),
            created_utc DATETIME,
            url TEXT,
            keywords TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            post_id VARCHAR(20),
            url TEXT,
            ocr_text TEXT,
            UNIQUE KEY(post_id, url(255)),
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def mask_author(author):
    return "user_" + str(hash(author) % 100000) if author else "unknown"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\.\,\-\_']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_keywords(text):
    words = [w.lower() for w in text.split() if w.lower() not in STOPWORDS]
    return ",".join(sorted(set(words))[:10])

def ocr_image(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        text = pytesseract.image_to_string(img)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""

def fetch_posts(subreddit_name):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    posts_data = []
    subreddit = reddit.subreddit(subreddit_name)
    
    # Try multiple sorting methods to get more posts
    sorting_methods = [
        ('new', subreddit.new),
        ('hot', subreddit.hot), 
        ('top', lambda **kwargs: subreddit.top(time_filter='all', **kwargs)),
        ('rising', subreddit.rising)
    ]
    
    seen_ids = set()
    
    for sort_name, sort_method in sorting_methods:
        print(f"Fetching from {sort_name}...")
        try:
            for submission in sort_method(limit=None):  # Reddit caps at ~1000 per method
                if submission.id in seen_ids:
                    continue
                    
                seen_ids.add(submission.id)
                
                title = clean_text(submission.title)
                body = clean_text(submission.selftext)
                author = mask_author(str(submission.author) if submission.author else None)
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                url = submission.url
                keywords = extract_keywords(f"{title} {body}")

                images = []
                if url.lower().endswith((".jpg", ".jpeg", ".png")):
                    ocr_text = ocr_image(url)
                    if ocr_text:
                        images.append((url, ocr_text))

                posts_data.append({
                    "id": submission.id,
                    "title": title,
                    "body": body,
                    "author": author,
                    "created": created,
                    "url": url,
                    "keywords": keywords,
                    "images": images
                })

            
            print(f"Total unique posts so far: {len(posts_data)}")
                
            # Rate limiting between methods
            time.sleep(2)
            
        except Exception as e:
            print(f"Error with {sort_name}: {e}")
            continue
    
    return posts_data[:n]

def save_posts(posts):
    conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME)
    cur = conn.cursor()

    post_rows = []
    image_rows = []
    for p in posts:
        post_rows.append((p["id"], p["title"], p["body"], p["author"], p["created"], p["url"], p["keywords"]))
        for img_url, ocr_text in p["images"]:
            image_rows.append((p["id"], img_url, ocr_text))

    # Batch insert posts
    cur.executemany("""
        INSERT INTO posts (id, title, body, author_masked, created_utc, url, keywords)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            title=VALUES(title),
            body=VALUES(body),
            author_masked=VALUES(author_masked),
            created_utc=VALUES(created_utc),
            url=VALUES(url),
            keywords=VALUES(keywords)
    """, post_rows)

    # Batch insert images (only with OCR text)
    if image_rows:
        cur.executemany("""
            INSERT INTO images (post_id, url, ocr_text)
            VALUES (%s,%s,%s)
            ON DUPLICATE KEY UPDATE ocr_text=VALUES(ocr_text)
        """, image_rows)

    conn.commit()
    print(f"Saved {len(posts)} posts and {len(image_rows)} images to database")
    cur.close()
    conn.close()

def main():
    posts = fetch_posts("Vacations")
    save_posts(posts)

if __name__ == "__main__":
    init_db()
    main()