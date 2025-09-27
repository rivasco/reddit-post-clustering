from reddit_scraper import init_db, main
import os
import mysql.connector
import uuid
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import chromadb
nltk.download('punkt')
nltk.download('punkt_tab')

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

def train_doc2vec(data):
    # preproces the documents, and create TaggedDocuments
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(data)]

    # train the Doc2vec model
    model = Doc2Vec(vector_size=100, min_count=1, epochs=100)

    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # get the document vectors
    document_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in data]
    
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
        include=["ids", "documents", "embeddings"]
    )
    ids = result["ids"]
    documents = result["documents"]
    embeddings = result["embeddings"]

    ######## PUT CLUSTERING CODE HERE


if __name__=='__main__':
    # fetch data
    init_db()
    main()

    # read data from MySQL
    rows = load_raw_texts()
    data = [
        " ".join(filter(None, [
            to_text(row.get("title")),
            to_text(row.get("keywords")),
            to_text(row.get("body")),
            to_text(row.get("images")),
        ])).strip()
        for row in rows
    ]

    model, document_vectors = train_doc2vec(data)
    store_document_vectors(data, document_vectors)
    document_clustering()

DB_USER=cindy
DB_PASS=NewP@ssw0rd
REDDIT_CLIENT_ID=SuXHKB3DJ3F7qFtGbsONeQ
REDDIT_CLIENT_SECRET=AmsSh2ZHJRJDC4uRvCWaO4KJ5RZy8Q
REDDIT_USER_AGENT=reddit-scraper-lab
