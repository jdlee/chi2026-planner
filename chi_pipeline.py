"""CHI 2026 Paper Pipeline: Scrape → Classify → Output"""

import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import yaml
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"


def load_themes() -> dict:
    """Load seed themes from config."""
    config_path = CONFIG_DIR / "themes.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def classify_papers(papers: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    """Classify papers against seed themes using Claude.

    Returns (classified_papers, all_themes).
    Processes papers in batches to manage API costs.
    """
    client = anthropic.Anthropic()
    seed_themes = config["seed_themes"]
    threshold = config.get("relevance_threshold", 0.3)
    max_emergent = config.get("max_emergent_themes", 10)

    theme_names = [t["name"] for t in seed_themes]
    theme_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}" for t in seed_themes
    )

    classified = []
    emergent_suggestions = []

    # Process in batches of 10
    batch_size = 10
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]

        papers_text = "\n---\n".join(
            f"Paper {j+1}:\nTitle: {p['title']}\nAbstract: {p['abstract']}"
            for j, p in enumerate(batch)
        )

        prompt = f"""You are classifying academic papers from CHI 2026 against research themes.

SEED THEMES:
{theme_descriptions}

PAPERS:
{papers_text}

For each paper, respond with JSON (no other text):
{{
  "papers": [
    {{
      "index": 1,
      "theme_scores": {{"Theme Name": 0.0-1.0, ...}},
      "emergent_themes": ["theme1", "theme2"]
    }},
    ...
  ]
}}

Rules:
- Score each paper against ALL seed themes (0.0 = irrelevant, 1.0 = perfect match)
- Suggest up to 2 emergent themes per paper that are NOT in the seed list
- Emergent themes should be specific enough to be useful (not "Technology" or "Design")
- Be calibrated: most papers should score < 0.3 on most themes
"""

        response = client.messages.create(
            model="claude-sonnet-4-6-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        result = json.loads(response.content[0].text)

        for j, paper_result in enumerate(result["papers"]):
            paper = batch[j].copy()
            paper["theme_scores"] = paper_result["theme_scores"]
            classified.append(paper)
            emergent_suggestions.extend(paper_result.get("emergent_themes", []))

        logger.info(f"Classified papers {i+1}-{i+len(batch)} of {len(papers)}")

    # Deduplicate and rank emergent themes
    emergent_counts = Counter(emergent_suggestions)
    top_emergent = [
        {"name": theme, "description": f"Emergent theme (appeared {count} times)", "emergent": True}
        for theme, count in emergent_counts.most_common(max_emergent)
        if count >= 2  # require at least 2 papers to suggest same theme
    ]

    all_themes = seed_themes + top_emergent

    return classified, all_themes


def save_classified(papers: list[dict], themes: list[dict]) -> None:
    """Save classified data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    (DATA_DIR / "chi2026_classified.json").write_text(json.dumps(papers, indent=2))
    (DATA_DIR / "themes.json").write_text(json.dumps(themes, indent=2))
    logger.info(f"Saved {len(papers)} classified papers and {len(themes)} themes")


def run_classify_only():
    """Classify papers from existing raw data."""
    raw_path = DATA_DIR / "chi2026_raw.json"
    if not raw_path.exists():
        logger.error(f"No raw data found at {raw_path}. Run scraper first.")
        return

    papers = json.loads(raw_path.read_text())
    config = load_themes()
    classified, themes = classify_papers(papers, config)
    save_classified(classified, themes)


if __name__ == "__main__":
    run_classify_only()
