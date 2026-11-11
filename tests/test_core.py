import time
from pathlib import Path

import pandas as pd
import pytest

from core import batch
from core.batch import BatchRunner, process_row
from core.providers import DemoProvider
from core.sheet import load_table, normalize, shots_for
from core.video import duration, ffmpeg_ok

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_ok(), reason="FFmpeg not installed")


def test_aliases_defaults_and_cleaning():
    df = pd.DataFrame({"Title": ["A topic", "", "Another"], "Length": ["45s", "", "12"], "Ratio": ["4:3", "", "1:1"]})
    jobs, mapping = normalize(df)
    assert mapping["topic"] == "Title" and len(jobs) == 2
    assert jobs[0]["duration"] == "30" and jobs[0]["aspect"] == "9:16"  # clamped, invalid ratio replaced
    assert jobs[1]["duration"] == "12" and jobs[1]["aspect"] == "1:1" and jobs[1]["voice"] == "coral"


def test_missing_topic_column_raises():
    with pytest.raises(ValueError):
        normalize(pd.DataFrame({"foo": ["x"]}))


def test_shot_count_and_ids_are_stable():
    jobs, _ = normalize(load_table("templates/sample.csv"))
    again, _ = normalize(load_table("templates/sample.csv"))
    assert [j["id"] for j in jobs] == [j["id"] for j in again] and len(jobs) == 5
    assert shots_for({"duration": "10"}) == 2 and shots_for({"duration": "30"}) == 6


def _job(topic="Demo topic", duration_s="10"):
    jobs, _ = normalize(pd.DataFrame({"topic": [topic], "duration": [duration_s], "aspect": ["1:1"]}))
    return jobs[0]


@needs_ffmpeg
def test_row_end_to_end_and_cache(tmp_path):
    row, stages = _job(), []
    res = process_row(row, DemoProvider(), None, tmp_path, lambda s, p: stages.append(s))
    assert res["state"] == "done" and duration(res["video"]) > 5
    assert (Path(res["video"]).parent / "subtitles.srt").exists() and "Animating" in stages
    assert process_row(row, DemoProvider(), None, tmp_path)["state"] == "cached"


@needs_ffmpeg
def test_ai_video_failure_falls_back_to_free_engine(tmp_path):
    class Broken:
        def animate(self, *args, **kwargs):
            raise RuntimeError("service down")

    res = process_row(_job("Fallback topic"), DemoProvider(), Broken(), tmp_path)
    assert res["state"] == "done" and res["fallbacks"] == 2


@needs_ffmpeg
def test_runner_isolates_failures_and_writes_manifest(tmp_path):
    class Flaky(DemoProvider):
        def plan(self, row, n):
            if "boom" in row["topic"]:
                raise RuntimeError("planner exploded")
            return super().plan(row, n)

    jobs, _ = normalize(pd.DataFrame({"topic": ["Good one", "boom"], "duration": ["10", "10"], "aspect": ["1:1", "1:1"]}))
    runner = BatchRunner(jobs, Flaky(), None, tmp_path / "lib", tmp_path / "batches", workers=2)
    runner.start()
    while runner.running:
        time.sleep(0.2)
    states = [runner.status[j["id"]]["state"] for j in jobs]
    assert states == ["done", "failed"] and runner.manifest_path.exists()
    assert "planner exploded" in runner.manifest()["error"].iloc[1]
    assert batch.process_row  # public API stays importable
