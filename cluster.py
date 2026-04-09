"""Cluster CHI papers using TF-IDF + UMAP + HDBSCAN and extract topic labels."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def build_corpus(papers: list[dict]) -> list[str]:
    """Combine title and abstract into a single document per paper."""
    docs = []
    for p in papers:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        doc = f"{title}. {title}. {abstract}" if abstract else f"{title}. {title}"
        docs.append(doc)
    return docs


def fit_tfidf(docs: list[str], max_features: int = 5000) -> tuple:
    """Fit TF-IDF vectorizer and return matrix + vectorizer."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    logger.info(f"TF-IDF matrix: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def run_umap(tfidf_matrix, n_neighbors: int = 30, min_dist: float = 0.1) -> np.ndarray:
    """Reduce TF-IDF to 2D with UMAP."""
    n_neighbors = min(n_neighbors, tfidf_matrix.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(tfidf_matrix)
    logger.info(f"UMAP complete: {coords.shape}")
    return coords


def run_clustering(tfidf_matrix, min_cluster_size: int = None) -> np.ndarray:
    """Cluster papers with HDBSCAN in TF-IDF space."""
    n_papers = tfidf_matrix.shape[0]
    if min_cluster_size is None:
        # Scale with dataset: min 5 for small sets, 15 for large
        min_cluster_size = max(5, min(15, n_papers // 20))
    n_neighbors = min(30, n_papers - 1)
    n_components = min(20, n_papers - 2)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(tfidf_matrix)
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
    return labels


def extract_topic_labels(
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    labels: np.ndarray,
    top_n_terms: int = 3,
) -> dict:
    """Extract top TF-IDF terms per cluster as human-readable topic labels."""
    feature_names = vectorizer.get_feature_names_out()
    cluster_ids = sorted(set(labels))
    topics = {}

    for cid in cluster_ids:
        if cid == -1:
            topics[-1] = {"label": "Uncategorized", "terms": []}
            continue

        mask = labels == cid
        cluster_tfidf = tfidf_matrix[mask].mean(axis=0)
        cluster_array = np.asarray(cluster_tfidf).flatten()
        top_indices = cluster_array.argsort()[-top_n_terms:][::-1]
        terms = [feature_names[i] for i in top_indices]

        label = " / ".join(t.title() for t in terms)
        topics[cid] = {"label": label, "terms": terms}
        logger.info(f"  Cluster {cid}: {label} ({mask.sum()} papers)")

    return topics


def compute_topic_scores(
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    topics: dict,
) -> pd.DataFrame:
    """Score each paper against each topic using mean TF-IDF of topic terms."""
    feature_names = list(vectorizer.get_feature_names_out())
    scores = {}

    for cid, topic in topics.items():
        if cid == -1:
            continue
        term_indices = []
        for term in topic["terms"]:
            if term in feature_names:
                term_indices.append(feature_names.index(term))

        if term_indices:
            topic_tfidf = tfidf_matrix[:, term_indices].mean(axis=1)
            col_values = np.asarray(topic_tfidf).flatten()
            max_val = col_values.max()
            if max_val > 0:
                col_values = col_values / max_val
            scores[topic["label"]] = col_values

    return pd.DataFrame(scores)


def run_full_clustering(papers: list[dict]) -> dict:
    """Run the full clustering pipeline."""
    docs = build_corpus(papers)
    tfidf_matrix, vectorizer = fit_tfidf(docs)
    coords = run_umap(tfidf_matrix)
    labels = run_clustering(tfidf_matrix)
    topics = extract_topic_labels(tfidf_matrix, vectorizer, labels)
    scores_df = compute_topic_scores(tfidf_matrix, vectorizer, topics)

    topic_names = list(scores_df.columns)

    for i, paper in enumerate(papers):
        paper["umap_x"] = float(coords[i, 0])
        paper["umap_y"] = float(coords[i, 1])
        paper["cluster"] = int(labels[i])
        paper["cluster_label"] = topics[int(labels[i])]["label"]
        paper["topic_scores"] = {col: float(scores_df.iloc[i][col]) for col in topic_names}

    return {
        "papers": papers,
        "topics": topics,
        "topic_names": topic_names,
    }


def save_clustered(result: dict) -> None:
    """Save clustered data to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "chi2026_clustered.json").write_text(
        json.dumps(result["papers"], indent=2)
    )
    # Convert int64 keys to str for JSON serialization
    topics_serializable = {str(k): v for k, v in result["topics"].items()}
    (DATA_DIR / "topics.json").write_text(
        json.dumps({"topics": topics_serializable, "topic_names": result["topic_names"]}, indent=2)
    )
    logger.info(
        f"Saved {len(result['papers'])} clustered papers "
        f"with {len(result['topic_names'])} topics"
    )


def main():
    """Load raw papers, cluster, save results."""
    raw_path = DATA_DIR / "chi2026_raw.json"
    if not raw_path.exists():
        logger.error(f"No raw data at {raw_path}. Run scraper first.")
        return

    papers = json.loads(raw_path.read_text())
    logger.info(f"Loaded {len(papers)} papers")

    result = run_full_clustering(papers)
    save_clustered(result)


if __name__ == "__main__":
    main()
