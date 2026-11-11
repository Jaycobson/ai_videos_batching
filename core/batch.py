"""Batch engine: one isolated job per sheet row, with caching, live status and a manifest."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .sheet import ASPECTS, shots_for
from .video import assemble, duration, fit_clip, living_clip, write_srt


def process_row(row, provider, motion, out_root, report=lambda stage, pct: None, reuse=True):
    """Plan -> images -> voice -> motion -> assemble. `motion` is a local AI video model or None (free living image)."""
    d = Path(out_root) / row["id"]
    video = d / "video.mp4"
    if reuse and video.exists():
        report("Cached", 1.0)
        return {"state": "cached", "video": str(video), "fallbacks": 0}
    d.mkdir(parents=True, exist_ok=True)
    w, h, size = ASPECTS[row["aspect"]]
    report("Planning", 0.05)
    plan = provider.plan(row, shots_for(row))
    (d / "plan.json").write_text(json.dumps(plan, indent=2))
    shots, n = plan["shots"], len(plan["shots"])

    images = []
    for i, s in enumerate(shots):
        prompt = (f"{s['image_prompt']} Recurring character: {plan['character']}. Visual style: {row['style']}. "
                  "Natural candid composition, people clearly visible, no text or lettering.")
        path = d / f"shot_{i + 1}.png"
        path.write_bytes(provider.image(prompt, size))
        images.append(path)
        report("Creating visuals", 0.08 + 0.32 * (i + 1) / n)

    voices = []
    for i, s in enumerate(shots):
        path = d / f"voice_{i + 1}.mp3"
        provider.speak(s["narration"], path, row["voice"], row["language"])
        voices.append(path)
        report("Recording voice", 0.40 + 0.15 * (i + 1) / n)

    durs = [max(3.0, duration(v) + 0.5) for v in voices]
    clips, fallbacks = [], 0
    for i, s in enumerate(shots):
        clip = d / f"clip_{i + 1}.mp4"
        try:
            if motion is None:
                raise LookupError("free engine")
            raw = d / f"raw_{i + 1}.mp4"
            motion.animate(images[i].read_bytes(), s["motion_prompt"], raw, size=(w, h))
            fit_clip(raw, clip, durs[i], (w, h))
        except Exception as e:  # local AI video failed: fall back to the free engine so the row still finishes
            fallbacks += motion is not None and not isinstance(e, LookupError)
            living_clip(images[i], clip, durs[i], (w, h), i)
        clips.append(clip)
        report("Animating", 0.55 + 0.35 * (i + 1) / n)

    report("Assembling", 0.93)
    assemble(clips, voices, video, durs)
    write_srt([s["narration"] for s in shots], durs, d / "subtitles.srt")
    report("Done", 1.0)
    return {"state": "done", "video": str(video), "fallbacks": fallbacks}


class BatchRunner:
    """Runs rows in parallel on a background thread; the UI polls `status`."""

    def __init__(self, rows, provider, motion, out_root, manifest_dir, workers=2, reuse=True):
        self.rows, self.provider, self.motion, self.reuse = rows, provider, motion, reuse
        self.out_root, self.manifest_dir, self.workers = Path(out_root), Path(manifest_dir), workers
        self.status = {r["id"]: {"state": "queued", "stage": "Queued", "pct": 0.0, "video": "", "error": "", "fallbacks": 0}
                       for r in rows}
        self.thread, self.started, self.finished, self.manifest_path = None, None, None, None

    @property
    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self):
        self.started = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _job(self, row):
        st = self.status[row["id"]]
        st.update(state="running", stage="Starting")
        try:
            res = process_row(row, self.provider, self.motion, self.out_root,
                              lambda stage, pct: st.update(stage=stage, pct=pct), self.reuse)
            st.update(state=res["state"], stage="Cached" if res["state"] == "cached" else "Done", pct=1.0,
                      video=res["video"], fallbacks=res["fallbacks"])
        except Exception as e:
            st.update(state="failed", stage="Failed", error=str(e)[:300])

    def _run(self):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self._job, self.rows))
        self.finished = time.time()
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.manifest_dir / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.manifest().to_csv(self.manifest_path, index=False)

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame([{"topic": r["topic"], "status": self.status[r["id"]]["state"], "video": self.status[r["id"]]["video"],
                              "ai_fallbacks": self.status[r["id"]]["fallbacks"], "error": self.status[r["id"]]["error"]}
                             for r in self.rows])
