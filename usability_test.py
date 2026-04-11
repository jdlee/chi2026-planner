"""Synthetic usability test for CHI 2026 Planner using Playwright + Claude API."""

import asyncio
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

RESULTS_DIR = Path("usability_test_results")
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
APP_URL = "http://localhost:8503"
MAX_STEPS = 12
MODEL = "claude-sonnet-4-20250514"

# --- Personas and Tasks ---

PERSONAS = [
    {
        "name": "Dr. Chen",
        "bio": (
            "Senior HCI researcher attending her 10th CHI. She studies VR interaction "
            "techniques and wants to build a focused schedule for Wednesday."
        ),
        "tasks": [
            {
                "id": "chen_t1",
                "description": (
                    "Find VR-related papers scheduled for Wednesday afternoon "
                    "and add 3 to your agenda."
                ),
            },
            {
                "id": "chen_t2",
                "description": (
                    "Use the topic hierarchy to explore what macro themes exist, "
                    "then drill into the one most relevant to VR."
                ),
            },
        ],
    },
    {
        "name": "Marcus",
        "bio": (
            "First-time PhD student attendee. Studies AI-assisted creativity tools. "
            "Overwhelmed by 2500+ papers and looking for a way to narrow down."
        ),
        "tasks": [
            {
                "id": "marcus_t1",
                "description": (
                    "Use the topic hierarchy to find papers about AI and creativity, "
                    "then select two that interest you."
                ),
            },
            {
                "id": "marcus_t2",
                "description": "Export your selections as a calendar file.",
            },
        ],
    },
    {
        "name": "Prof. Adler",
        "bio": (
            "Lab manager building a shared agenda for her research group. Wants to "
            "find award-winning papers efficiently and share them with the team."
        ),
        "tasks": [
            {
                "id": "adler_t1",
                "description": (
                    "Filter the papers table to show only Best Paper award winners."
                ),
            },
            {
                "id": "adler_t2",
                "description": (
                    "Save your selection and download a Markdown agenda for the lab."
                ),
            },
        ],
    },
    {
        "name": "Keiko",
        "bio": (
            "Accessibility researcher focused on assistive technologies for blind "
            "and deaf communities. Wants comprehensive coverage across all days."
        ),
        "tasks": [
            {
                "id": "keiko_t1",
                "description": (
                    "Find papers about accessibility, blindness, or deaf communities "
                    "across all conference days."
                ),
            },
            {
                "id": "keiko_t2",
                "description": (
                    "Narrow your view to only Wednesday and Thursday papers using "
                    "the schedule controls in the sidebar."
                ),
            },
        ],
    },
]


ANALYST_SYSTEM = """\
You are {name}, {bio}

You are performing a usability test of a conference paper planning web application.
You will see screenshots of the application at each step.

For each screenshot, respond in EXACTLY this JSON format:
{{
  "think_aloud": "Your honest first-person observation about what you see and what's confusing or clear",
  "expectation": "What you expect will happen when you take your action",
  "action": {{
    "type": "click" | "scroll" | "type" | "check" | "uncheck" | "done" | "stuck",
    "target": "natural language description of the element to interact with",
    "text": "text to type (only for type actions)",
    "direction": "up" | "down" (only for scroll actions)
  }},
  "task_status": "in_progress" | "complete" | "stuck"
}}

Rules:
- Be honest about confusion. If a label is unclear, say so.
- If you can't figure out how to proceed after looking carefully, set task_status to "stuck".
- When the task is done, set task_status to "complete" and action type to "done".
- Describe click targets by their visible text or position, e.g. "the Topic Hierarchy expander",
  "the checkbox in the AM row under Wed", "the Award column filter icon".
- You can only see what's on screen. If you need to scroll to find something, say so.
- Be specific about what you find confusing or well-designed.
"""


async def take_screenshot(page, name: str) -> Path:
    """Take a full-page screenshot and save it."""
    path = SCREENSHOTS_DIR / f"{name}.png"
    await page.screenshot(path=str(path), full_page=False)
    return path


def screenshot_to_base64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


async def ask_analyst(
    client: anthropic.Anthropic,
    persona: dict,
    task: str,
    screenshot_path: Path,
    history: list[dict],
) -> dict:
    """Send screenshot to Claude and get think-aloud response."""
    system = ANALYST_SYSTEM.format(**persona)

    messages = []

    # Build conversation history (abbreviated)
    if history:
        history_text = "Steps taken so far:\n"
        for i, h in enumerate(history):
            history_text += f"  {i+1}. {h['action_desc']} → {h['outcome']}\n"
        messages.append({"role": "user", "content": history_text})
        messages.append({"role": "assistant", "content": "Understood. Show me the current screen."})

    # Current screenshot + task
    img_data = screenshot_to_base64(screenshot_path)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_data},
            },
            {
                "type": "text",
                "text": f"Current task: {task}\n\nWhat do you see? What would you do next?",
            },
        ],
    })

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )

    text = response.content[0].text

    # Parse JSON from response (handle markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {
            "think_aloud": text,
            "expectation": "Could not parse response",
            "action": {"type": "stuck", "target": ""},
            "task_status": "stuck",
        }


async def execute_action(page, action: dict) -> str:
    """Execute an action described by the analyst. Returns outcome description."""
    action_type = action.get("type", "stuck")
    target = action.get("target", "")

    if action_type == "done":
        return "Task marked complete"

    if action_type == "stuck":
        return "User is stuck"

    if action_type == "scroll":
        direction = action.get("direction", "down")
        delta = -500 if direction == "up" else 500
        await page.mouse.wheel(0, delta)
        await page.wait_for_timeout(1000)
        return f"Scrolled {direction}"

    if action_type == "type":
        text = action.get("text", "")
        # Find visible input elements
        inputs = page.locator("input:visible")
        count = await inputs.count()
        for i in range(count):
            inp = inputs.nth(i)
            placeholder = await inp.get_attribute("placeholder") or ""
            aria = await inp.get_attribute("aria-label") or ""
            if target.lower() in placeholder.lower() or target.lower() in aria.lower():
                await inp.fill(text)
                await inp.press("Enter")
                await page.wait_for_timeout(1500)
                return f"Typed '{text}' in {target}"
        # Fallback: type in first visible input
        if count > 0:
            await inputs.first.fill(text)
            await inputs.first.press("Enter")
            await page.wait_for_timeout(1500)
            return f"Typed '{text}' in first input (fallback)"
        return "No input found"

    if action_type in ("click", "check", "uncheck"):
        # Strategy: try multiple selector approaches
        selectors_to_try = []

        # 1. Exact text match
        selectors_to_try.append(f"text='{target}'")
        selectors_to_try.append(f"text=\"{target}\"")

        # 2. Partial text match via role
        selectors_to_try.append(f"button:has-text('{target}')")
        selectors_to_try.append(f"[role='button']:has-text('{target}')")

        # 3. For expanders/details
        if "expander" in target.lower() or "hierarchy" in target.lower():
            selectors_to_try.append("details summary:has-text('Topic Hierarchy')")
            selectors_to_try.append("details summary:has-text('Topic')")

        # 4. For checkboxes
        if action_type in ("check", "uncheck") or "checkbox" in target.lower():
            selectors_to_try.append(f"label:has-text('{target}') input[type='checkbox']")
            selectors_to_try.append(f"label:has-text('{target}')")

        # 5. Broader text search
        words = target.split()
        if len(words) >= 2:
            key_phrase = " ".join(words[:3])
            selectors_to_try.append(f"*:has-text('{key_phrase}')")

        # 6. For sidebar elements
        if "sidebar" in target.lower() or any(
            w in target.lower() for w in ["am", "pm", "schedule", "save", "clear", "export"]
        ):
            for sel in list(selectors_to_try):
                selectors_to_try.append(f"[data-testid='stSidebar'] {sel}")

        # 7. For AgGrid filter buttons
        if "filter" in target.lower() or "award" in target.lower():
            selectors_to_try.append(".ag-header-cell:has-text('Award') .ag-icon-filter")
            selectors_to_try.append(".ag-header-cell:has-text('Award') button")
            selectors_to_try.append(".ag-floating-filter-input")

        # 8. Download buttons
        if "download" in target.lower() or "markdown" in target.lower() or "calendar" in target.lower():
            selectors_to_try.append("button:has-text('Markdown')")
            selectors_to_try.append("button:has-text('Calendar')")
            selectors_to_try.append("button:has-text('.ics')")

        for selector in selectors_to_try:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=500):
                    await loc.click(timeout=2000)
                    await page.wait_for_timeout(2000)
                    return f"Clicked: {target} (via {selector})"
            except Exception:
                continue

        # Last resort: try clicking at text position
        try:
            loc = page.get_by_text(target, exact=False).first
            if await loc.is_visible(timeout=500):
                await loc.click(timeout=2000)
                await page.wait_for_timeout(2000)
                return f"Clicked text: {target}"
        except Exception:
            pass

        return f"Could not find element: {target}"

    return f"Unknown action type: {action_type}"


async def reset_app(page):
    """Navigate to app and reset state for a fresh task."""
    await page.goto(APP_URL, wait_until="networkidle", timeout=30000)
    await page.wait_for_timeout(3000)


async def run_task(
    page,
    client: anthropic.Anthropic,
    persona: dict,
    task: dict,
) -> dict:
    """Run a single task for a persona. Returns task result dict."""
    task_id = task["id"]
    task_desc = task["description"]
    steps = []
    history = []

    print(f"    Task: {task_desc[:60]}...")

    for step_num in range(1, MAX_STEPS + 1):
        # Screenshot
        ss_name = f"{task_id}_s{step_num:02d}"
        ss_path = await take_screenshot(page, ss_name)

        # Ask Claude
        response = await ask_analyst(client, persona, task_desc, ss_path, history)

        think_aloud = response.get("think_aloud", "")
        expectation = response.get("expectation", "")
        action = response.get("action", {"type": "stuck", "target": ""})
        status = response.get("task_status", "in_progress")

        # Execute action
        if status in ("complete", "stuck") or action.get("type") in ("done", "stuck"):
            outcome = "Task ended"
        else:
            outcome = await execute_action(page, action)

        action_desc = f"{action.get('type', '?')}: {action.get('target', '?')}"

        step_record = {
            "step": step_num,
            "screenshot": f"screenshots/{ss_name}.png",
            "think_aloud": think_aloud,
            "expectation": expectation,
            "action": action,
            "action_desc": action_desc,
            "outcome": outcome,
            "task_status": status,
        }
        steps.append(step_record)
        history.append({"action_desc": action_desc, "outcome": outcome})

        print(f"      Step {step_num}: {action_desc} → {outcome}")

        if status in ("complete", "stuck"):
            break

    final_status = steps[-1]["task_status"] if steps else "unknown"

    # Identify confusion points
    confusion = [
        s["think_aloud"]
        for s in steps
        if any(w in s["think_aloud"].lower() for w in [
            "confus", "unclear", "not sure", "don't see", "can't find",
            "expected", "surprising", "weird", "where", "how do i",
            "doesn't seem", "lost",
        ])
    ]

    return {
        "task_id": task_id,
        "task_description": task_desc,
        "persona": persona["name"],
        "status": final_status,
        "steps": steps,
        "n_steps": len(steps),
        "confusion_points": confusion,
    }


def generate_report(all_results: list[dict]) -> str:
    """Generate the Markdown usability test report."""
    lines = [
        "# Synthetic Usability Test Report",
        f"*CHI 2026 Planner — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
    ]

    # --- Summary ---
    lines.append("## Summary\n")
    total_tasks = len(all_results)
    completed = sum(1 for r in all_results if r["status"] == "complete")
    stuck = sum(1 for r in all_results if r["status"] == "stuck")
    avg_steps = sum(r["n_steps"] for r in all_results) / max(total_tasks, 1)

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Tasks | {total_tasks} |")
    lines.append(f"| Completed | {completed} |")
    lines.append(f"| Stuck | {stuck} |")
    lines.append(f"| Avg steps | {avg_steps:.1f} |")
    lines.append("")

    # Aggregate confusion points
    all_confusion = []
    for r in all_results:
        for c in r["confusion_points"]:
            all_confusion.append((r["persona"], c))

    if all_confusion:
        lines.append("### Top Usability Issues\n")
        for persona, quote in all_confusion:
            lines.append(f"- **{persona}:** {quote}\n")
        lines.append("")

    # --- Per-persona task details ---
    current_persona = None
    for result in all_results:
        if result["persona"] != current_persona:
            current_persona = result["persona"]
            persona_data = next(
                p for p in PERSONAS if p["name"] == current_persona
            )
            lines.append(f"---\n## {current_persona}\n")
            lines.append(f"*{persona_data['bio']}*\n")

        status_icon = {"complete": "PASS", "stuck": "STUCK", "in_progress": "TIMEOUT"}
        s = status_icon.get(result["status"], result["status"])
        lines.append(
            f"### Task: {result['task_description']}\n"
        )
        lines.append(f"**Result:** {s} in {result['n_steps']} steps\n")

        for step in result["steps"]:
            lines.append(f"#### Step {step['step']}\n")
            lines.append(f"![step {step['step']}]({step['screenshot']})\n")
            lines.append(f"> {step['think_aloud']}\n")
            lines.append(f"**Expected:** {step['expectation']}\n")
            lines.append(
                f"**Action:** {step['action_desc']}  \n"
                f"**Outcome:** {step['outcome']}\n"
            )

        if result["confusion_points"]:
            lines.append("**Confusion points:**\n")
            for c in result["confusion_points"]:
                lines.append(f"- {c}")
            lines.append("")

    return "\n".join(lines)


async def main():
    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    all_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--window-size=1440,900"])
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        for persona in PERSONAS:
            print(f"\n{'='*60}")
            print(f"Persona: {persona['name']}")
            print(f"{'='*60}")

            for task in persona["tasks"]:
                await reset_app(page)
                result = await run_task(page, client, persona, task)
                all_results.append(result)
                print(f"    → {result['status']} ({result['n_steps']} steps)")

        await browser.close()

    # Generate report
    report = generate_report(all_results)
    report_path = RESULTS_DIR / "report.md"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # Also save raw JSON
    json_path = RESULTS_DIR / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
