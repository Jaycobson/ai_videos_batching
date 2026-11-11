from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.video import ffmpeg_ok

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_loads_and_reads_sample_sheet():
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    next(b for b in at.button if "sample" in b.label).click()
    at.run(timeout=30)
    assert not at.exception and len(at.session_state["jobs"]) == 5


@pytest.mark.skipif(not ffmpeg_ok(), reason="FFmpeg not installed")
def test_demo_batch_runs_from_the_ui():
    at = AppTest.from_file(APP).run(timeout=30)
    next(b for b in at.button if "sample" in b.label).click()
    at.run(timeout=30)
    at.session_state["jobs"] = at.session_state["jobs"][:1]
    at.radio(key="engine").set_value("🧪 Demo mode · no API needed")
    next(b for b in at.button if "Start batch" in b.label).click()
    at.run(timeout=180)
    assert not at.exception
    runner = at.session_state["runner"]
    assert [s["state"] for s in runner.status.values()] in (["done"], ["cached"])
