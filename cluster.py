"""Cluster CHI papers using TF-IDF + UMAP + Spectral Clustering with 3-level hierarchy.

Hierarchy:
  Level 1 (Macro):  ~8-12 broad themes  (agglomerative on L2 centroids)
  Level 2 (Mid):    ~20-30 topic groups  (agglomerative on L3 centroids)
  Level 3 (Fine):   ~40-60 sub-topics    (Spectral Clustering on UMAP embedding)

Labels use lemmatized terms with parent-differentiating selection at each level.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import nltk
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# Ensure NLTK data is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

_lemmatizer = WordNetLemmatizer()

# Known acronyms that should stay uppercased
_ACRONYMS = {
    "ai", "vr", "ar", "xr", "mr", "hci", "ui", "ux", "llm", "xai",
    "chi", "blv", "nfc", "adhd", "asd", "ocd", "dhh", "ehmi", "av",
    "hri", "iot", "api", "3d", "2d", "fdm", "genai", "hmd",
}


def _format_label_term(term: str) -> str:
    """Title-case a term, preserving known acronyms as uppercase."""
    words = term.split()
    formatted = []
    for w in words:
        if w.lower() in _ACRONYMS:
            formatted.append(w.upper())
        else:
            formatted.append(w.capitalize())
    return " ".join(formatted)


def _lemmatize_term(term: str) -> str:
    """Lemmatize a term (single word or bigram), deduplicating lemma forms."""
    words = term.split()
    lemmas = []
    seen = set()
    for w in words:
        lemma = _lemmatizer.lemmatize(w.lower())
        if lemma not in seen:
            lemmas.append(lemma)
            seen.add(lemma)
    return " ".join(lemmas)


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


def run_fine_clustering(tfidf_matrix, n_clusters: int = 60) -> np.ndarray:
    """Level 3: Fine-grained clusters with Spectral Clustering.

    Reduces TF-IDF to moderate dimensions with UMAP first, then applies
    spectral clustering with nearest-neighbor affinity for balanced clusters.
    Every paper gets assigned — no noise points.
    """
    n_papers = tfidf_matrix.shape[0]
    n_clusters = min(n_clusters, n_papers - 1)

    # Reduce to moderate dimensions for spectral clustering
    n_components = min(30, n_papers - 2)
    n_neighbors = min(30, n_papers - 1)
    logger.info(f"UMAP reduction to {n_components}D for spectral clustering...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(tfidf_matrix)

    logger.info(f"Running spectral clustering with {n_clusters} clusters...")
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=15,
        assign_labels="kmeans",
        random_state=42,
        n_init=10,
    )
    labels = clusterer.fit_predict(embedding)
    actual_clusters = len(set(labels))

    # Report cluster size distribution
    sizes = np.bincount(labels)
    logger.info(
        f"Level 3 (fine): {actual_clusters} clusters, 0 noise points | "
        f"sizes: min={sizes.min()}, median={int(np.median(sizes))}, max={sizes.max()}"
    )
    return labels


def compute_centroids(tfidf_matrix, labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Compute mean TF-IDF centroid for each cluster (excluding noise=-1)."""
    cluster_ids = sorted(set(labels) - {-1})
    centroids = []
    for cid in cluster_ids:
        mask = labels == cid
        centroid = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        centroids.append(centroid)
    return np.array(centroids), cluster_ids


def build_hierarchy(
    tfidf_matrix,
    fine_labels: np.ndarray,
    n_mid: int = None,
    n_macro: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build levels 2 and 1 by agglomerative clustering of fine-cluster centroids.

    Returns (mid_labels, macro_labels) arrays aligned to the original paper indices.
    """
    centroids, fine_ids = compute_centroids(tfidf_matrix, fine_labels)
    n_fine = len(fine_ids)

    if n_mid is None:
        n_mid = max(8, n_fine // 2)
    if n_macro is None:
        n_macro = min(12, max(9, n_mid // 3))

    n_mid = min(n_mid, n_fine)
    n_macro = min(n_macro, n_mid)

    logger.info(f"Hierarchy target: {n_macro} macro -> {n_mid} mid -> {n_fine} fine")

    # Use UMAP to reduce centroid dimensions, then Ward linkage for balanced splits
    n_centroid_components = min(10, n_fine - 2)
    centroid_reducer = umap.UMAP(
        n_components=n_centroid_components,
        n_neighbors=min(15, n_fine - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    centroid_embedding = centroid_reducer.fit_transform(centroids)

    # Level 2: group fine clusters into mid-level topics (Ward = balanced)
    agg_mid = AgglomerativeClustering(n_clusters=n_mid, linkage="ward")
    fine_to_mid = agg_mid.fit_predict(centroid_embedding)

    # Level 1: group mid clusters into macro themes
    mid_ids = sorted(set(fine_to_mid))
    mid_centroids = []
    for mid_id in mid_ids:
        fine_mask = fine_to_mid == mid_id
        mid_centroid = centroid_embedding[fine_mask].mean(axis=0)
        mid_centroids.append(mid_centroid)
    mid_centroids = np.array(mid_centroids)

    agg_macro = AgglomerativeClustering(n_clusters=n_macro, linkage="ward")
    mid_to_macro = agg_macro.fit_predict(mid_centroids)

    # Map back to paper level
    fine_id_to_idx = {cid: i for i, cid in enumerate(fine_ids)}
    mid_labels = np.full(len(fine_labels), -1, dtype=int)
    macro_labels = np.full(len(fine_labels), -1, dtype=int)

    for i, fl in enumerate(fine_labels):
        if fl == -1:
            continue
        fidx = fine_id_to_idx[fl]
        mid_id = fine_to_mid[fidx]
        mid_labels[i] = mid_id
        macro_labels[i] = mid_to_macro[mid_id]

    logger.info(
        f"Hierarchy built: {n_macro} macro, {len(set(mid_labels) - {-1})} mid, "
        f"{len(set(fine_labels) - {-1})} fine"
    )
    return mid_labels, macro_labels


def _select_distinctive_terms(
    tfidf_matrix,
    feature_names: np.ndarray,
    cluster_mask: np.ndarray,
    parent_mask: np.ndarray | None,
    top_n: int = 3,
    exclude_lemmas: set[str] | None = None,
) -> list[str]:
    """Select top TF-IDF terms that distinguish a cluster from its parent.

    For the top level (no parent), selects terms that distinguish from the corpus.
    For child levels, selects terms where the cluster's mean TF-IDF most exceeds
    the parent's mean TF-IDF — i.e., what makes this child distinctive.
    Deduplicates by lemma form to avoid "game" and "games" appearing together.
    """
    if exclude_lemmas is None:
        exclude_lemmas = set()

    cluster_tfidf = np.asarray(tfidf_matrix[cluster_mask].mean(axis=0)).flatten()

    if parent_mask is not None:
        parent_tfidf = np.asarray(tfidf_matrix[parent_mask].mean(axis=0)).flatten()
        # Score = how much this cluster exceeds its parent, normalized
        parent_max = parent_tfidf.max()
        if parent_max > 0:
            parent_norm = parent_tfidf / parent_max
        else:
            parent_norm = parent_tfidf
        cluster_max = cluster_tfidf.max()
        if cluster_max > 0:
            cluster_norm = cluster_tfidf / cluster_max
        else:
            cluster_norm = cluster_tfidf
        # Distinctiveness: high in cluster, low in parent
        scores = cluster_norm - 0.5 * parent_norm
    else:
        scores = cluster_tfidf

    ranked_indices = scores.argsort()[::-1]

    terms = []
    seen_lemmas = set(exclude_lemmas)
    for idx in ranked_indices:
        if len(terms) >= top_n:
            break
        raw_term = feature_names[idx]
        lemma = _lemmatize_term(raw_term)
        # Skip if any lemma word overlaps with already-seen lemmas
        lemma_words = set(lemma.split())
        if lemma_words & seen_lemmas:
            continue
        seen_lemmas.update(lemma_words)
        terms.append(lemma)

    return terms


def extract_hierarchical_labels(
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    macro_labels: np.ndarray,
    mid_labels: np.ndarray,
    fine_labels: np.ndarray,
    top_n_terms: int = 3,
) -> tuple[dict, dict, dict]:
    """Extract labels at all three levels with parent-differentiating term selection.

    Macro: top terms vs corpus
    Mid: top terms that differentiate from parent macro cluster
    Fine: top terms that differentiate from parent mid cluster

    Terms are lemmatized and deduplicated. Lower levels exclude parent label terms
    to minimize overlap across hierarchy levels.
    """
    feature_names = vectorizer.get_feature_names_out()

    # --- Macro labels (top level, no parent) ---
    macro_topics = {}
    macro_term_lemmas = {}  # macro_id -> set of lemma words used
    for cid in sorted(set(macro_labels)):
        if cid == -1:
            macro_topics[-1] = {"label": "Uncategorized", "terms": [], "count": int((macro_labels == -1).sum())}
            macro_term_lemmas[-1] = set()
            continue
        mask = macro_labels == cid
        terms = _select_distinctive_terms(tfidf_matrix, feature_names, mask, None, top_n_terms)
        label = " – ".join(_format_label_term(t) for t in terms)
        count = int(mask.sum())
        macro_topics[cid] = {"label": label, "terms": terms, "count": count}
        macro_term_lemmas[cid] = set()
        for t in terms:
            macro_term_lemmas[cid].update(t.split())
        logger.info(f"  Macro {cid}: {label} ({count} papers)")

    # --- Mid labels (differentiate from parent macro) ---
    mid_topics = {}
    mid_term_lemmas = {}
    for cid in sorted(set(mid_labels)):
        if cid == -1:
            mid_topics[-1] = {"label": "Uncategorized", "terms": [], "count": int((mid_labels == -1).sum())}
            mid_term_lemmas[-1] = set()
            continue
        mid_mask = mid_labels == cid
        # Find parent macro
        sample_idx = np.where(mid_mask)[0][0]
        macro_id = int(macro_labels[sample_idx])
        parent_mask = macro_labels == macro_id
        parent_lemmas = macro_term_lemmas.get(macro_id, set())
        terms = _select_distinctive_terms(
            tfidf_matrix, feature_names, mid_mask, parent_mask, top_n_terms,
            exclude_lemmas=parent_lemmas,
        )
        label = " – ".join(_format_label_term(t) for t in terms)
        count = int(mid_mask.sum())
        mid_topics[cid] = {"label": label, "terms": terms, "count": count}
        mid_term_lemmas[cid] = set(parent_lemmas)
        for t in terms:
            mid_term_lemmas[cid].update(t.split())
        logger.info(f"  Mid {cid}: {label} ({count} papers)")

    # --- Fine labels (differentiate from parent mid) ---
    fine_topics = {}
    for cid in sorted(set(fine_labels)):
        if cid == -1:
            fine_topics[-1] = {"label": "Uncategorized", "terms": [], "count": int((fine_labels == -1).sum())}
            continue
        fine_mask = fine_labels == cid
        # Find parent mid
        sample_idx = np.where(fine_mask)[0][0]
        mid_id = int(mid_labels[sample_idx])
        parent_mask = mid_labels == mid_id
        parent_lemmas = mid_term_lemmas.get(mid_id, set())
        terms = _select_distinctive_terms(
            tfidf_matrix, feature_names, fine_mask, parent_mask, top_n_terms,
            exclude_lemmas=parent_lemmas,
        )
        label = " – ".join(_format_label_term(t) for t in terms)
        count = int(fine_mask.sum())
        fine_topics[cid] = {"label": label, "terms": terms, "count": count}
        logger.info(f"  Fine {cid}: {label} ({count} papers)")

    return macro_topics, mid_topics, fine_topics


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
            # Match lemmatized terms back to feature names
            for fi, fname in enumerate(feature_names):
                if _lemmatize_term(fname) == term:
                    term_indices.append(fi)
                    break

        if term_indices:
            topic_tfidf = tfidf_matrix[:, term_indices].mean(axis=1)
            col_values = np.asarray(topic_tfidf).flatten()
            max_val = col_values.max()
            if max_val > 0:
                col_values = col_values / max_val
            scores[topic["label"]] = col_values

    return pd.DataFrame(scores)


def run_full_clustering(papers: list[dict]) -> dict:
    """Run the full 3-level hierarchical clustering pipeline."""
    docs = build_corpus(papers)
    tfidf_matrix, vectorizer = fit_tfidf(docs)
    coords = run_umap(tfidf_matrix)

    # Level 3: fine-grained spectral clusters
    fine_labels = run_fine_clustering(tfidf_matrix)

    # Levels 2 & 1: agglomerative hierarchy over fine centroids
    mid_labels, macro_labels = build_hierarchy(tfidf_matrix, fine_labels, n_macro=10)

    # Extract hierarchical labels with parent differentiation + lemmatization
    macro_topics, mid_topics, fine_topics = extract_hierarchical_labels(
        tfidf_matrix, vectorizer, macro_labels, mid_labels, fine_labels, top_n_terms=3
    )

    # Topic scores based on fine-level topics
    scores_df = compute_topic_scores(tfidf_matrix, vectorizer, fine_topics)
    topic_names = list(scores_df.columns)

    # Build hierarchy map: which fine clusters belong to which mid/macro
    hierarchy = {}
    for cid in sorted(set(fine_labels) - {-1}):
        mask = fine_labels == cid
        paper_idx = np.where(mask)[0][0]
        mid_id = int(mid_labels[paper_idx])
        macro_id = int(macro_labels[paper_idx])
        hierarchy[int(cid)] = {"mid": mid_id, "macro": macro_id}

    # Annotate papers
    for i, paper in enumerate(papers):
        paper["umap_x"] = float(coords[i, 0])
        paper["umap_y"] = float(coords[i, 1])
        paper["cluster"] = int(fine_labels[i])
        paper["cluster_label"] = fine_topics[int(fine_labels[i])]["label"]
        paper["mid_cluster"] = int(mid_labels[i])
        paper["mid_label"] = mid_topics[int(mid_labels[i])]["label"]
        paper["macro_cluster"] = int(macro_labels[i])
        paper["macro_label"] = macro_topics[int(macro_labels[i])]["label"]
        paper["topic_scores"] = {col: float(scores_df.iloc[i][col]) for col in topic_names}

    return {
        "papers": papers,
        "topics": {
            "macro": {str(k): v for k, v in macro_topics.items()},
            "mid": {str(k): v for k, v in mid_topics.items()},
            "fine": {str(k): v for k, v in fine_topics.items()},
        },
        "hierarchy": {str(k): v for k, v in hierarchy.items()},
        "topic_names": topic_names,
    }


def save_clustered(result: dict) -> None:
    """Save clustered data to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "chi2026_clustered.json").write_text(
        json.dumps(result["papers"], indent=2)
    )
    (DATA_DIR / "topics.json").write_text(
        json.dumps({
            "topics": result["topics"],
            "hierarchy": result["hierarchy"],
            "topic_names": result["topic_names"],
        }, indent=2)
    )
    logger.info(
        f"Saved {len(result['papers'])} clustered papers with 3-level hierarchy: "
        f"{len(result['topics']['macro']) - 1} macro, "
        f"{len(result['topics']['mid']) - 1} mid, "
        f"{len(result['topics']['fine']) - 1} fine"
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
