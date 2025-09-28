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
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


# Chromadb Configuration
COLLECTION_NAME = "Vacation_Texts_and_Vectors"
client = chromadb.Client()


# MySQL Configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "phpmyadmin")
DB_PASS = os.getenv("DB_PASS", "root")
DB_NAME = os.getenv("DB_NAME", "reddit_scraper")
TABLE_NAME = "posts"


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
    cur.execute(f"SELECT * FROM {TABLE_NAME}")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def clustering(document_vectors):
    # normalization and PCA
    Xn = normalize(document_vectors)                        
    pca = PCA(n_components=min(50, Xn.shape[1]), random_state=42)
    Xp = pca.fit_transform(Xn)
    
    # Apply K-Means clustering
    best_k = 2
    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    labels_k = list(km.fit_predict(Xp))

    dump(pca, 'pca.joblib')
    dump(km, 'kmeans.joblib')

    return km, pca, labels_k

def train_doc2vec(data):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    # get the document vectors
    document_vectors = model.encode(data, convert_to_numpy=True)

    return model, document_vectors


def infer_vectors(text):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    vector = model.encode(text, convert_to_numpy=True)
    return np.asanyarray(vector, dtype="float32")


def store_document_vectors(documents, document_vectors):
    # delete old collection if exists
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(name=COLLECTION_NAME)

    # clustering
    km, pca, labels = clustering(document_vectors)

    # create a collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "store raw texts and embeddings of vacation reddit posts"}
    )

    # write raw texts and vectors
    ids = [f"doc-{uuid.uuid4()}" for _ in range(len(documents))]
    metadatas = [{"source": "Reddit", "cluster": int(label)} for label in labels]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=document_vectors.tolist(),
        metadatas=metadatas
    )


def store_new_vector(text, vector, km, pca):
    pca = load('pca.joblib')
    km = load('kmeans.joblib')
    v = vector.reshape(1, -1)
    v_n = normalize(v)
    v_p = pca.transform(v_n)
    cid = int(km.predict(v_p)[0])

    # create a collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # write new text and vector
    new_id = f"doc-{uuid.uuid4()}"
    metadata = {"source": "Reddit", "cluster": cid}

    collection.add(
        ids=[new_id],
        documents=[text],
        embeddings=[vector],
        metadatas=[metadata]
    )
    print(f"New message belongs to cluster {cid}.")

    return new_id, cid


def run_pipeline_once():
    try:
        logging.info("Fetching data (init_db/main) ...")
        init_db()
        main()
        logging.info("Fetching done.")
    except Exception as e:
        logging.exception("Fetching/processing failed: %s", e)
        return

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
    except Exception as e:
        logging.exception("Database updates or vectorization failed: %s", e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run pipeline once (--interval) and/or add a new message (--message).")
    parser.add_argument("--interval", type=float, default=None,
                        help="If provided, run the pipeline once.")
    parser.add_argument("--message", type=str, default=None,
                        help="If provided, treat it as a new post to encode and store.")
    parser.add_argument("--log-level", default="INFO", help="Log level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    did_anything = False

    # A) run pipeline once if interval is provided
    if args.interval is not None:
        logging.info("Running pipeline once (interval=%.2f provided).", args.interval)
        run_pipeline_once()
        did_anything = True

    # B) if message is provided, encode & insert with persisted PCA/KMeans
    if args.message:
        logging.info("Processing single message insert.")
        # load / create the sentence model (stateless; not saved to disk)
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        vec = sbert.encode(args.message, convert_to_numpy=True).astype("float32")
        try:
            new_id, cid = store_new_vector(args.message, vec, km=None, pca=None)  # will load from disk
            logging.info("Inserted new message id=%s, cluster=%d", new_id, cid)
        except Exception as e:
            logging.exception("Failed to insert new message: %s", e)
        did_anything = True

    if not did_anything:
        parser.print_help()
