"""Quick smoke test: classify a handful of fake papers."""
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from chi_pipeline import classify_papers, load_themes

SAMPLE_PAPERS = [
    {
        "title": "Understanding Trust Calibration in AI-Assisted Decision Making",
        "authors": "Smith, A.; Jones, B.",
        "abstract": "We study how users calibrate trust when AI systems provide recommendations with varying accuracy. Our experiment with 200 participants reveals that explanation style significantly impacts trust calibration.",
        "session": "Trust and AI",
        "time": "9:00-10:15",
        "location": "Room 101",
        "date": "2026-04-27",
        "start_time": "09:00",
        "end_time": "10:15",
    },
    {
        "title": "Conversational Repair Strategies in LLM-Powered Chatbots",
        "authors": "Lee, C.; Park, D.",
        "abstract": "We analyze how users recover from conversational breakdowns when interacting with large language model chatbots. Through a diary study, we identify five repair strategies.",
        "session": "LLM Experiences",
        "time": "10:30-11:45",
        "location": "Room 205",
        "date": "2026-04-27",
        "start_time": "10:30",
        "end_time": "11:45",
    },
]


def test_classify_sample():
    """Run classification on sample papers and verify structure."""
    config = load_themes()
    classified, themes = classify_papers(SAMPLE_PAPERS, config)

    assert len(classified) == 2
    assert "theme_scores" in classified[0]
    assert all(
        theme["name"] in classified[0]["theme_scores"]
        for theme in config["seed_themes"]
    )
    for score in classified[0]["theme_scores"].values():
        assert 0.0 <= score <= 1.0

    print("Classification test passed!")
    print(json.dumps(classified[0]["theme_scores"], indent=2))


if __name__ == "__main__":
    test_classify_sample()
