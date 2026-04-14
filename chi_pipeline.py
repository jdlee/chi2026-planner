"""CHI 2026 Paper Pipeline: Scrape -> Classify -> Output"""

import json
import re
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import yaml
import anthropic
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv(Path(__file__).parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"



def _parse_time_range(text: str) -> tuple[str, str]:
    """Parse '11:15 AM - 12:45 PM' into ('11:15', '12:45') in 24h format."""
    parts = text.split(" - ")
    times = []
    for part in parts:
        part = part.strip()
        m = re.match(r"(\d+):(\d+)\s*(AM|PM)", part)
        if m:
            h, mi, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
            if ampm == "PM" and h != 12:
                h += 12
            elif ampm == "AM" and h == 12:
                h = 0
            times.append(f"{h:02d}:{mi:02d}")
    start = times[0] if len(times) >= 1 else ""
    end = times[1] if len(times) >= 2 else ""
    return start, end


async def _scrape_day_page(page, date_str: str) -> list[dict]:
    """Scrape all papers from a single day's program page.

    Assumes the page is already navigated to the day URL and loaded.
    """
    papers = []

    # Scroll to load all timeslots on this day
    prev_height = 0
    for _ in range(10):
        curr_height = await page.evaluate("document.body.scrollHeight")
        if curr_height == prev_height:
            break
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1500)
        prev_height = curr_height
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(500)

    timeslots = await page.query_selector_all("div.timeslot")
    logger.info(f"  {date_str}: found {len(timeslots)} timeslots")

    for timeslot in timeslots:
        # Scroll timeslot into view to trigger lazy loading
        await timeslot.scroll_into_view_if_needed()
        await page.wait_for_timeout(2000)

        time_el = await timeslot.query_selector("h3.timeslot-time")
        time_text = (await time_el.inner_text()).strip() if time_el else ""
        slot_start, slot_end = _parse_time_range(time_text)

        session_cards = await timeslot.query_selector_all("session-card")

        for card in session_cards:
            name_el = await card.query_selector("span.name")
            session_name = (await name_el.inner_text()).strip() if name_el else "Unknown"

            type_el = await card.query_selector("span.type-name")
            session_type = (await type_el.inner_text()).strip() if type_el else ""

            room_el = await card.query_selector("session-room-data span[translate]")
            location = (await room_el.inner_text()).strip() if room_el else ""

            count_el = await card.query_selector("contents-quantity")
            count_text = (await count_el.inner_text()).strip() if count_el else ""
            item_match = re.match(r"(\d+)\s+item", count_text)
            item_count = int(item_match.group(1)) if item_match else 0

            if item_count == 0:
                continue

            expand_btn = await card.query_selector("button.icon-btn-toggle-card")
            if not expand_btn:
                continue

            try:
                await expand_btn.click()
                await page.wait_for_timeout(1500)

                item_cards = await card.query_selector_all(
                    "item-card, content-item, .item-card"
                )
                if not item_cards:
                    item_cards = await card.query_selector_all(
                        "a.link-block:not(.session-card-header)"
                    )

                for item in item_cards:
                    title_el = await item.query_selector(
                        "span.name, h4.card-data-name, .card-data-name span.name"
                    )
                    title = (await title_el.inner_text()).strip() if title_el else ""

                    authors_el = await item.query_selector(
                        "person-list, .people-container"
                    )
                    authors = (await authors_el.inner_text()).strip() if authors_el else ""
                    authors = re.sub(r"\s*,\s*", ", ", authors).strip(", ")

                    # Get content URL for abstract fetching
                    link_el = await item.query_selector("a[href*='/program/content/']")
                    if not link_el:
                        link_el = item if await item.get_attribute("href") else None
                    content_url = ""
                    if link_el:
                        href = await link_el.get_attribute("href")
                        if href:
                            content_url = href

                    # Check for award badges (Best Paper, Honorable Mention)
                    award = ""
                    award_el = await item.query_selector("award-label")
                    if award_el:
                        award = (await award_el.inner_text()).strip()

                    if title:
                        papers.append({
                            "title": title,
                            "authors": authors,
                            "abstract": "",
                            "award": award,
                            "content_url": content_url,
                            "session": session_name,
                            "session_type": session_type,
                            "time": time_text,
                            "location": location,
                            "date": date_str,
                            "start_time": slot_start,
                            "end_time": slot_end,
                        })

                await expand_btn.click()
                await page.wait_for_timeout(300)

            except Exception as e:
                logger.warning(f"Failed to expand session '{session_name}': {e}")

        logger.info(
            f"  {time_text}: {len(session_cards)} sessions, {len(papers)} papers so far"
        )

    return papers


async def scrape_chi_program(url: str = "https://programs.sigchi.org/chi/2026") -> list[dict]:
    """Scrape CHI 2026 program using Playwright.

    Navigates to each day's page individually to avoid Angular's virtual
    scrolling issues on the 'all' index page.
    """
    # Day URL slugs for CHI 2026 (April 13-17)
    day_pages = [
        ("13-apr", "2026-04-13"),
        ("14-apr", "2026-04-14"),
        ("15-apr", "2026-04-15"),
        ("16-apr", "2026-04-16"),
        ("17-apr", "2026-04-17"),
    ]

    all_papers = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for day_slug, date_str in day_pages:
            day_url = f"{url.rstrip('/')}/program/{day_slug}"
            logger.info(f"Navigating to {day_url}")
            await page.goto(day_url, wait_until="networkidle", timeout=60000)
            await page.wait_for_timeout(3000)

            day_papers = await _scrape_day_page(page, date_str)
            all_papers.extend(day_papers)
            logger.info(f"Day {date_str}: {len(day_papers)} papers ({len(all_papers)} total)")

        await browser.close()

    logger.info(f"Scraped {len(all_papers)} papers total across {len(day_pages)} days")
    return all_papers


async def fetch_abstracts(
    papers: list[dict],
    base_url: str = "https://programs.sigchi.org",
) -> list[dict]:
    """Fetch abstracts by visiting each paper's detail page. Resumable.

    Skips papers that already have an abstract or have no content_url.
    Saves progress every 100 papers.
    """
    to_fetch = [
        (i, p) for i, p in enumerate(papers)
        if not p.get("abstract") and p.get("content_url")
    ]
    if not to_fetch:
        logger.info("All papers already have abstracts (or no URLs). Nothing to fetch.")
        return papers

    logger.info(f"Fetching abstracts for {len(to_fetch)} papers...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for count, (idx, paper) in enumerate(to_fetch):
            url = base_url + paper["content_url"]
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1000)

                # Abstract is in a <p> after <h4>Abstract</h4> inside white-block
                abstract_el = await page.query_selector("white-block p")
                if abstract_el:
                    abstract = (await abstract_el.inner_text()).strip()
                    papers[idx]["abstract"] = abstract

            except Exception as e:
                logger.warning(f"Failed to fetch abstract for '{paper['title'][:50]}': {e}")

            if (count + 1) % 50 == 0:
                logger.info(f"  Fetched {count + 1}/{len(to_fetch)} abstracts")

            # Save progress every 100 papers
            if (count + 1) % 100 == 0:
                save_raw(papers)
                logger.info(f"  Progress saved at {count + 1} abstracts")

        await browser.close()

    fetched = sum(1 for p in papers if p.get("abstract"))
    logger.info(f"Abstracts fetched: {fetched}/{len(papers)} papers now have abstracts")
    return papers


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Merge papers that appear in multiple timeslots into single entries.

    CHI schedules posters/demos across multiple sessions. We keep one entry
    per unique content_url, merging all scheduled appearances into a
    'schedule' list. For papers without a content_url, we deduplicate by title.
    The primary (first) appearance's session info stays in the top-level fields.
    """
    from collections import OrderedDict

    seen = OrderedDict()  # key -> merged paper
    title_to_key = {}  # title -> first key, for cross-URL dedup
    for p in papers:
        key = p.get("content_url") or p.get("title", "")
        if not key:
            seen[id(p)] = p
            continue

        # Also check if we've seen this title under a different content_url
        title = p.get("title", "")
        if key not in seen and title in title_to_key:
            key = title_to_key[title]

        if key not in seen:
            if title:
                title_to_key[title] = key
            merged = dict(p)
            merged["schedule"] = [{
                "session": p.get("session", ""),
                "session_type": p.get("session_type", ""),
                "date": p.get("date", ""),
                "time": p.get("time", ""),
                "start_time": p.get("start_time", ""),
                "end_time": p.get("end_time", ""),
                "location": p.get("location", ""),
            }]
            seen[key] = merged
        else:
            merged = seen[key]
            merged["schedule"].append({
                "session": p.get("session", ""),
                "session_type": p.get("session_type", ""),
                "date": p.get("date", ""),
                "time": p.get("time", ""),
                "start_time": p.get("start_time", ""),
                "end_time": p.get("end_time", ""),
                "location": p.get("location", ""),
            })
            # Keep the abstract if the existing one is empty
            if not merged.get("abstract") and p.get("abstract"):
                merged["abstract"] = p["abstract"]
            # Keep the award if the existing one is empty
            if not merged.get("award") and p.get("award"):
                merged["award"] = p["award"]

    result = list(seen.values())
    n_removed = len(papers) - len(result)
    if n_removed:
        logger.info(f"Deduplicated: {len(papers)} -> {len(result)} papers ({n_removed} duplicate appearances merged)")
    return result


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
    papers = await fetch_abstracts(papers)
    papers = deduplicate_papers(papers)
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
    max_emergent = config.get("max_emergent_themes", 10)

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
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response, handling possible preamble text
        response_text = response.content[0].text
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                logger.error(f"Failed to parse classification response for batch {i}")
                continue

        for j, paper_result in enumerate(result.get("papers", [])):
            paper = batch[j].copy()
            paper["theme_scores"] = paper_result["theme_scores"]
            classified.append(paper)
            emergent_suggestions.extend(paper_result.get("emergent_themes", []))

        logger.info(f"Classified papers {i+1}-{i+len(batch)} of {len(papers)}")

        # Save progress every 50 batches
        if len(classified) % 500 < batch_size:
            _save_progress(classified, seed_themes, emergent_suggestions, max_emergent)

    # Final save
    _save_progress(classified, seed_themes, emergent_suggestions, max_emergent)

    # Deduplicate and rank emergent themes
    emergent_counts = Counter(emergent_suggestions)
    top_emergent = [
        {"name": theme, "description": f"Emergent theme (appeared {count} times)", "emergent": True}
        for theme, count in emergent_counts.most_common(max_emergent)
        if count >= 2  # require at least 2 papers to suggest same theme
    ]

    all_themes = seed_themes + top_emergent

    return classified, all_themes


def _save_progress(classified, seed_themes, emergent_suggestions, max_emergent):
    """Save intermediate classification progress."""
    emergent_counts = Counter(emergent_suggestions)
    top_emergent = [
        {"name": theme, "description": f"Emergent theme (appeared {count} times)", "emergent": True}
        for theme, count in emergent_counts.most_common(max_emergent)
        if count >= 2
    ]
    all_themes = seed_themes + top_emergent
    save_classified(classified, all_themes)
    logger.info(f"Progress saved: {len(classified)} papers classified")


def save_classified(papers: list[dict], themes: list[dict]) -> None:
    """Save classified data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    (DATA_DIR / "chi2026_classified.json").write_text(json.dumps(papers, indent=2))
    (DATA_DIR / "themes.json").write_text(json.dumps(themes, indent=2))
    logger.info(f"Saved {len(papers)} classified papers and {len(themes)} themes")


def run_classify_only():
    """Classify papers from existing raw data. Supports resume."""
    raw_path = DATA_DIR / "chi2026_raw.json"
    if not raw_path.exists():
        logger.error(f"No raw data found at {raw_path}. Run scraper first.")
        return

    papers = json.loads(raw_path.read_text())
    config = load_themes()

    # Resume: check if partial classification exists
    classified_path = DATA_DIR / "chi2026_classified.json"
    already_done = 0
    prior_classified = []
    if classified_path.exists():
        prior_classified = json.loads(classified_path.read_text())
        already_done = len(prior_classified)
        if already_done >= len(papers):
            logger.info("All papers already classified. Nothing to do.")
            return
        if already_done > 0:
            logger.info(f"Resuming: {already_done} papers already classified, {len(papers) - already_done} remaining")

    remaining = papers[already_done:]
    new_classified, themes = classify_papers(remaining, config)

    # Merge prior + new
    all_classified = prior_classified + new_classified
    save_classified(all_classified, themes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHI 2026 Paper Pipeline")
    parser.add_argument(
        "mode",
        choices=["scrape", "abstracts", "dedup", "classify", "cluster", "full"],
        default="full",
        nargs="?",
        help="Pipeline mode: scrape, abstracts, dedup, classify, cluster, or full",
    )
    args = parser.parse_args()

    if args.mode == "scrape":
        papers = asyncio.run(scrape_chi_program())
        save_raw(papers)
    elif args.mode == "abstracts":
        raw_path = DATA_DIR / "chi2026_raw.json"
        if not raw_path.exists():
            logger.error("No raw data. Run 'scrape' first.")
        else:
            papers = json.loads(raw_path.read_text())
            papers = asyncio.run(fetch_abstracts(papers))
            save_raw(papers)
    elif args.mode == "dedup":
        raw_path = DATA_DIR / "chi2026_raw.json"
        if not raw_path.exists():
            logger.error("No raw data. Run 'scrape' first.")
        else:
            papers = json.loads(raw_path.read_text())
            papers = deduplicate_papers(papers)
            save_raw(papers)
    elif args.mode == "classify":
        run_classify_only()
    elif args.mode == "cluster":
        from cluster import main as cluster_main
        cluster_main()
    else:
        asyncio.run(run_full_pipeline())
