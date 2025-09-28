
import sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from collections import defaultdict
import spacy
import umap.umap_ as umap
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Load embeddings
embeddings = np.load("vectors.npy") 

X = embeddings

# normalization and PCA
Xn = normalize(X)                        
Xp = PCA(n_components=min(50, Xn.shape[1]), random_state=42).fit_transform(Xn)

# pick k by silhouette (cosine)
best_k, best_score, best_labels = None, -1.0, None
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels_k = km.fit_predict(Xp)
    score = silhouette_score(Xp, labels_k, metric="cosine")
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels_k

labels = best_labels 
best_k, best_score

print(f"\nBest k = {best_k} (silhouette = {best_score:.4f})")
np.save("labels.npy", labels)

# read docs file
docs = []
with open("data.json", "r", encoding="utf-8") as f:
    for line in f:
        texts = json.loads(line)
        docs.append(texts)

# combines docs based on their cluster labels
docs_dict = defaultdict(list)
for i, j in zip(labels, docs):
    docs_dict[str(i)].append([j["title"], j["body"], j["keywords"]] + ([j["images"]] if len(j["images"]) != 0 else []))

# get common words of all messages in each cluster
keyword_dict = defaultdict(list)

for key, values in docs_dict.items():
    if len(values) == 0:
        continue

    cluster_text = [i for sublist in values for i in sublist]

    # load small English model in spaCy package to perform lemmatization
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    def lemmatize_text(text):
        doc = nlp(text)
        return " ".join([token.lemma_.lower() for token in doc if token.is_alpha])
    processed_text = [lemmatize_text(text) for text in cluster_text]

    # Compute TF-IDF within this cluster
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(cluster_text)

    # check if a word is numeric and non-year
    def is_pure_number(token):
        return token.isdigit() and len(token) < 4

    # Get top keywords (highest average TF-IDF score)
    avg_tfidf = np.asarray(X.mean(axis=0)).flatten()
    features = vectorizer.get_feature_names_out()
    top_indices = avg_tfidf.argsort()[::-1][:20]
    top_words = []
    for idx in top_indices:
        word = features[idx]
        if not is_pure_number(word):
            top_words.append(word)
        if len(top_words) == 15:
            break

    keyword_dict[key] = top_words

# plot the clusters and keywords
pio.renderers.default = "browser"

reducer = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42)
X2 = reducer.fit_transform(Xp)

fig = px.scatter(
    x=X2[:, 0], y=X2[:, 1],
    color=labels.astype(str),
    title="Clustering results of reddit posts",
    labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Cluster"}
)

G = X2.mean(axis=0)
pad_scale = 0.45
jitter_y = 0.06

for i, c in enumerate(np.unique(labels)):
    pts = X2[labels == c]
    if pts.size == 0:
        continue

    cx, cy = pts.mean(axis=0)
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()

    v = np.array([cx, cy]) - G
    if np.allclose(v, 0):
        v = np.array([1.0, 0.0])
    v = v / np.linalg.norm(v)

    x_lab = cx + pad_scale * w * v[0]
    y_lab = cy + pad_scale * h * v[1] + ((-1)**i) * jitter_y * h

    kws = (keyword_dict.get(str(c), []) or [])[:15]
    text = ", ".join(kws[:8]) if len(kws) <= 8 else f"{', '.join(kws[:8])}<br>{', '.join(kws[8:])}"

    fig.add_annotation(
        x=x_lab, y=y_lab,
        ax=cx, ay=cy,
        text=text if text else "(no keywords)",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
        arrowcolor="rgba(80,80,80,0.35)",
        bgcolor="rgba(255,255,255,0.45)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
        font=dict(size=12),
        align="center"
    )

fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
fig.write_html("clusters_with_keywords.html", include_plotlyjs="cdn", auto_open=True)


# Verify similarity inside each cluster by displaying content of top 5 closest posts
pac_n = normalize(Xp)
top_docs = {}

for c in np.unique(labels):
    idx = np.where(labels == c)[0]
    ckey = str(int(c))
    
    if ckey not in docs_dict or len(docs_dict[ckey]) == 0:
        continue

    assert len(docs_dict[ckey]) == len(idx), \
        f"docs_dict[{ckey}] length {len(docs_dict[ckey])} != number of docs in cluster {c}"

    centroid = normalize(pac_n[idx].mean(axis=0, keepdims=True))
    sims = (pac_n[idx] @ centroid.T).ravel()
    order = np.argsort(-sims)[:5] 

    sel = []
    for local_i in order:
        global_i = int(idx[local_i])
        text = docs_dict[ckey][local_i]
        sel.append((global_i, float(sims[local_i]), text))
    top_docs[ckey] = sel

for ckey in sorted(top_docs, key=lambda x: int(x)):
    print(f"\nCluster {ckey} — top 5 closest to centroid")
    for rank, (gi, score, text) in enumerate(top_docs[ckey], 1):
        preview = (text[:600] + "…") if len(text) > 600 else text
        print(preview)


