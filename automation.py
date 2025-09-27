from reddit_scraper import init_db, main
import os
import mysql.connector
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import chromadb
import argparse
import logging
import time

# Chromadb Configuration
COLLECTION_NAME = "Vacation_Texts_and_Vectors"
client = chromadb.Client()

# MySQL Configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "phpmyadmin")
DB_PASS = os.getenv("DB_PASS", "root")
DB_NAME = os.getenv("DB_NAME", "reddit_scraper")

def to_text(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple, set)):
        return " ".join(map(str, x))
    if isinstance(x, dict):
        return " ".join(f"{k}:{v}" for k, v in x.items())
    return str(x)


def load_raw_texts():
    # Read text from MySQL
    conn = mysql.connector.connect(host=DB_HOST, 
                                   port=DB_PORT, 
                                   user=DB_USER, 
                                   password=DB_PASS,
                                   database=DB_NAME)
    cur = conn.cursor(dictionary=True)
    cur.execute("select p.id as id, p.title as title, p.body as body, p.keywords as keywords, img_agg.ocr_concat as images from posts as p left join (select post_id, group_concat(ocr_text order by id separator ' ') as ocr_concat from images group by post_id) as img_agg on p.id = img_agg.post_id")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows

def train_doc2vec(data):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    # get the document vectors
    document_vectors = model.encode(data, convert_to_numpy=True)

    return model, document_vectors


def store_document_vectors(documents, document_vectors):
    # delete old collection if exists
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(name=COLLECTION_NAME)

    # create a collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "store raw texts and embeddings of vacation reddit posts"}
    )

    # write raw texts and vectors
    ids = [f"doc-{uuid.uuid4()}" for _ in range(len(documents))]
    metadatas = [{"source": "Reddit"} for _ in documents]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=document_vectors.tolist(),
        metadatas=metadatas
    )

def document_clustering():
    # get documents and vectors
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    result = collection.get(
        include=["documents", "embeddings"]
    )
    ids = result["ids"]
    documents = result["documents"]
    embeddings = result["embeddings"]

    ######## PUT CLUSTERING CODE HERE


def run_pipeline_once():  # NEW
    try:
        logging.info("Fetching data (init_db/main) ...")
        init_db()
        main()
        logging.info("Fetching done.")
    except Exception as e:
        logging.exception("Fetching/processing failed: %s", e)
        return  # 失败就结束本轮

    try:
        logging.info("Loading data from MySQL ...")
        rows = load_raw_texts()
        if not rows:
            logging.warning("Cannot read any data from MySQL .")
            return
        data = [
            " ".join(filter(None, [
                to_text(row.get("title")),
                to_text(row.get("keywords")),
                to_text(row.get("body")),
                to_text(row.get("images")),
            ])).strip()
            for row in rows
        ]
        logging.info("Processing data (Doc2Vec training) ...")
        model, document_vectors = train_doc2vec(data)
        logging.info("Processing done. Writing vectors to Chroma ...")
        store_document_vectors(data, document_vectors)
        logging.info("Chroma updated.")
        logging.info("Clustering ...")
        document_clustering()
        logging.info("Clustering done.")
    except Exception as e:
        logging.exception("Database updates or vectorization failed: %s", e)

if __name__ == '__main__':
    # NEW: Command-line arguments and logging
    parser = argparse.ArgumentParser(description="Run reddit pipeline and update DB on an interval (minutes).")
    parser.add_argument("interval", type=float,
                        help="Interval in minutes. For example, 5 means run every 5 minutes; 0 means run once and exit.")
    parser.add_argument("--log-level", default="INFO", help="Log level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    interval = max(0.0, args.interval)
    logging.info("Program started, interval=%.2f minutes.", interval)

    # Run once immediately
    run_pipeline_once()

    # If interval > 0, run repeatedly at the specified interval
    if interval > 0:
        try:
            while True:
                logging.info("Sleeping for %.2f minutes ...", interval)
                time.sleep(interval * 60.0)
                run_pipeline_once()
        except KeyboardInterrupt:
            logging.info("Received interrupt signal, exiting.")
