"""Export selected papers as Markdown agenda and .ics calendar."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event

# CHI 2026 is in Barcelona, Spain
VENUE_TZ = ZoneInfo("Europe/Madrid")


def generate_markdown(
    selected_papers: list[dict],
    themes: list[dict],
    theme_threshold: float = 0.3,
) -> str:
    """Generate a Markdown agenda document from selected papers."""
    lines = []
    lines.append("# CHI 2026 — My Agenda\n")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # Theme summary
    lines.append("## Themes of Interest\n")
    for theme in themes:
        count = sum(
            1 for p in selected_papers
            if p.get("theme_scores", {}).get(theme["name"], 0) >= theme_threshold
        )
        if count > 0:
            lines.append(f"- **{theme['name']}** — {count} paper{'s' if count != 1 else ''}")
    lines.append("")

    # Group by date
    by_date: dict[str, list[dict]] = {}
    for paper in selected_papers:
        date = paper.get("date", "Unknown Date")
        by_date.setdefault(date, []).append(paper)

    # Sort dates, then papers within each date by start_time
    for date in sorted(by_date.keys()):
        lines.append(f"## {date}\n")
        day_papers = sorted(by_date[date], key=lambda p: p.get("start_time", ""))

        # Check for conflicts
        for i, p1 in enumerate(day_papers):
            for p2 in day_papers[i + 1:]:
                if _times_overlap(p1, p2):
                    lines.append(
                        f"> **CONFLICT:** '{p1['title'][:40]}...' and "
                        f"'{p2['title'][:40]}...' overlap!\n"
                    )

        for paper in day_papers:
            time_str = paper.get("time", "")
            location = paper.get("location", "")
            tags = _format_theme_tags(paper, themes, theme_threshold)

            lines.append(f"### {time_str} — {location}\n")
            lines.append(f"**{paper['title']}**\n")
            lines.append(f"*{paper['authors']}*\n")
            if tags:
                lines.append(f"Themes: {tags}\n")
            abstract = paper.get("abstract", "")
            if abstract:
                # First two sentences as summary
                sentences = abstract.split(". ")
                summary = ". ".join(sentences[:2]) + ("." if len(sentences) > 1 else "")
                lines.append(f"{summary}\n")
            lines.append("---\n")

    return "\n".join(lines)


def generate_ics(
    selected_papers: list[dict],
    themes: list[dict],
    theme_threshold: float = 0.3,
    year: int = 2026,
) -> bytes:
    """Generate an .ics calendar file from selected papers."""
    cal = Calendar()
    cal.add("prodid", "-//CHI 2026 Agenda Planner//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", "CHI 2026 Agenda")

    for paper in selected_papers:
        event = Event()
        event.add("summary", paper["title"])

        # Parse date and time
        start_dt = _parse_datetime(paper, year)
        if start_dt:
            event.add("dtstart", start_dt)
            # Estimate 15 min per paper if no end time
            end_dt = _parse_end_datetime(paper, year) or (start_dt + timedelta(minutes=15))
            event.add("dtend", end_dt)

        if paper.get("location"):
            event.add("location", paper["location"])

        # Description: authors + abstract snippet + theme tags
        desc_parts = []
        if paper.get("authors"):
            desc_parts.append(f"Authors: {paper['authors']}")
        if paper.get("abstract"):
            desc_parts.append(f"\n{paper['abstract'][:300]}...")
        tags = _format_theme_tags(paper, themes, theme_threshold)
        if tags:
            desc_parts.append(f"\nThemes: {tags}")
        event.add("description", "\n".join(desc_parts))

        cal.add_component(event)

    return cal.to_ical()


def _times_overlap(p1: dict, p2: dict) -> bool:
    """Check if two papers' time slots overlap."""
    try:
        s1, e1 = p1.get("start_time", ""), p1.get("end_time", "")
        s2, e2 = p2.get("start_time", ""), p2.get("end_time", "")
        if not all([s1, e1, s2, e2]):
            return False
        return s1 < e2 and s2 < e1
    except (TypeError, ValueError):
        return False


def _format_theme_tags(paper: dict, themes: list[dict], threshold: float) -> str:
    """Format theme tags for a paper."""
    scores = paper.get("theme_scores", {})
    matching = [
        f"`{name}` ({score:.1f})"
        for name, score in sorted(scores.items(), key=lambda x: -x[1])
        if score >= threshold
    ]
    return ", ".join(matching)


def _parse_datetime(paper: dict, year: int) -> datetime | None:
    """Parse start datetime from paper metadata."""
    date_str = paper.get("date", "")
    time_str = paper.get("start_time", "")
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=VENUE_TZ)
    except ValueError:
        return None


def _parse_end_datetime(paper: dict, year: int) -> datetime | None:
    """Parse end datetime from paper metadata."""
    date_str = paper.get("date", "")
    time_str = paper.get("end_time", "")
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=VENUE_TZ)
    except ValueError:
        return None
