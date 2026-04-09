"""CHI 2026 Paper Pipeline: Scrape → Classify → Output"""

import json
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import yaml
import anthropic
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"


async def scrape_chi_program(url: str = "https://programs.sigchi.org/chi/2026") -> list[dict]:
    """Scrape CHI 2026 program using Playwright.

    NOTE: The CSS selectors below are initial guesses based on typical
    conference program site patterns. Run this once, inspect the output,
    and update selectors as needed. The site may also require clicking
    through navigation (days, tracks) to load all papers.
    """
    papers = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Navigating to {url}")
        await page.goto(url, wait_until="networkidle", timeout=60000)

        # Wait for content to render
        await page.wait_for_timeout(3000)

        # --- SELECTOR DISCOVERY MODE ---
        discovery_file = DATA_DIR / "page_dump.html"
        if not (DATA_DIR / "chi2026_raw.json").exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            html = await page.content()
            discovery_file.write_text(html)
            logger.info(f"Page HTML dumped to {discovery_file} for selector discovery")

        # --- PAPER EXTRACTION ---
        # These selectors are PLACEHOLDERS — update after inspecting page_dump.html
        session_elements = await page.query_selector_all("[class*='session'], [class*='Session']")
        logger.info(f"Found {len(session_elements)} session elements")

        for session_el in session_elements:
            session_title = await session_el.query_selector(
                "[class*='title'], [class*='name'], h2, h3"
            )
            session_name = await session_title.inner_text() if session_title else "Unknown Session"

            time_el = await session_el.query_selector("[class*='time'], [class*='date'], time")
            time_text = await time_el.inner_text() if time_el else ""

            location_el = await session_el.query_selector("[class*='room'], [class*='location']")
            location_text = await location_el.inner_text() if location_el else ""

            paper_elements = await session_el.query_selector_all(
                "[class*='paper'], [class*='item'], [class*='entry'], [class*='submission']"
            )

            for paper_el in paper_elements:
                title_el = await paper_el.query_selector("[class*='title'], h3, h4, a")
                title = await title_el.inner_text() if title_el else "Untitled"

                authors_el = await paper_el.query_selector("[class*='author']")
                authors = await authors_el.inner_text() if authors_el else ""

                abstract_el = await paper_el.query_selector("[class*='abstract'], [class*='description']")
                abstract = await abstract_el.inner_text() if abstract_el else ""

                paper_time_el = await paper_el.query_selector("[class*='time'], time")
                paper_time = await paper_time_el.inner_text() if paper_time_el else time_text

                papers.append({
                    "title": title.strip(),
                    "authors": authors.strip(),
                    "abstract": abstract.strip(),
                    "session": session_name.strip(),
                    "time": paper_time.strip(),
                    "location": location_text.strip(),
                    "date": "",
                    "start_time": "",
                    "end_time": "",
                })

        await browser.close()

    logger.info(f"Scraped {len(papers)} papers")
    return papers


def save_raw(papers: list[dict]) -> None:
    """Save raw scraped data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "chi2026_raw.json"
    out.write_text(json.dumps(papers, indent=2))
    logger.info(f"Saved {len(papers)} papers to {out}")


async def run_full_pipeline():
    """Run scrape + classify."""
    papers = await scrape_chi_program()
    save_raw(papers)

    config = load_themes()
    classified, themes = classify_papers(papers, config)
    save_classified(classified, themes)


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
    parser = argparse.ArgumentParser(description="CHI 2026 Paper Pipeline")
    parser.add_argument(
        "mode",
        choices=["scrape", "classify", "full"],
        default="full",
        nargs="?",
        help="Pipeline mode: scrape only, classify only, or full pipeline",
    )
    args = parser.parse_args()

    if args.mode == "scrape":
        asyncio.run(scrape_chi_program())
    elif args.mode == "classify":
        run_classify_only()
    else:
        asyncio.run(run_full_pipeline())
