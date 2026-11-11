"""Read a CSV/Excel sheet and turn each row into a clean, validated job."""
import hashlib
import json
import re

import pandas as pd

ALIASES = {
    "topic": ["topic", "title", "subject", "idea", "theme", "prompt"],
    "audience": ["audience", "target", "for"],
    "tone": ["tone", "mood"],
    "style": ["style", "visual_style", "visual", "look"],
    "duration": ["duration", "length", "seconds"],
    "aspect": ["aspect", "aspect_ratio", "ratio", "format"],
    "voice": ["voice"],
    "language": ["language", "lang"],
    "character": ["character", "characters", "person", "people", "cast"],
    "motion": ["motion", "action", "movement"],
    "cta": ["cta", "call_to_action", "closing"],
}
DEFAULTS = {"audience": "a general audience", "tone": "warm and engaging", "style": "cinematic, vibrant, realistic",
            "duration": "20", "aspect": "9:16", "voice": "coral", "language": "English", "character": "", "motion": "", "cta": ""}
# aspect -> (width, height, OpenAI image size)
ASPECTS = {"9:16": (720, 1280, "1024x1536"), "1:1": (720, 720, "1024x1024"), "16:9": (1280, 720, "1536x1024")}
MAX_ROWS = 25


def load_table(source, name=None) -> pd.DataFrame:
    name = (name or getattr(source, "name", str(source))).lower()
    df = pd.read_csv(source, dtype=str) if name.endswith(".csv") else pd.read_excel(source, dtype=str)
    return df.fillna("")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "video"


def shots_for(row: dict) -> int:
    return min(6, max(2, round(int(row["duration"]) / 5)))


def normalize(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Returns (jobs, column_mapping). Raises ValueError if there is no topic column."""
    cols = {re.sub(r"\s+", "_", str(c).strip().lower()): c for c in df.columns}
    mapping = {key: next((cols[a] for a in names if a in cols), None) for key, names in ALIASES.items()}
    if not mapping["topic"]:
        raise ValueError("Your sheet needs a 'topic' column (aliases: title, subject, idea, theme, prompt).")
    jobs = []
    for _, r in df.iterrows():
        topic = str(r[mapping["topic"]]).strip()
        if not topic:
            continue
        row = {"topic": topic}
        for key in ALIASES:
            if key != "topic":
                value = str(r[mapping[key]]).strip() if mapping[key] else ""
                row[key] = value or DEFAULTS[key]
        digits = re.sub(r"\D", "", row["duration"])
        row["duration"] = str(min(30, max(10, int(digits)))) if digits else DEFAULTS["duration"]
        row["aspect"] = row["aspect"] if row["aspect"] in ASPECTS else DEFAULTS["aspect"]
        fingerprint = hashlib.sha1(json.dumps(row, sort_keys=True).encode()).hexdigest()[:8]
        row["id"] = f"{slug(topic)}-{fingerprint}"
        jobs.append(row)
    return jobs[:MAX_ROWS], mapping
