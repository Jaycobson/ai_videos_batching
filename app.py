import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.batch import BatchRunner
from core.engines import build_engines, check_env, effective_workers, gpu_status
from core.sheet import MAX_ROWS, load_table, normalize, shots_for

load_dotenv()
LIB, BATCHES = Path("outputs/library"), Path("outputs/batches")
ENGINES = {"🪄 Living image · free camera motion": "living",
           "🧠 Local AI video · open source (SVD or LTX-Video, GPU)": "local",
           "🧪 Demo mode · no API needed": "demo"}
VIDEO_MODELS = {"LTX-Video (follows your motion prompt)": "ltx", "Stable Video Diffusion XT (animates the image)": "svd"}
IMAGE_SOURCES = {"OpenAI images (uses credits)": "openai", "Local SDXL-Turbo (open source, GPU)": "sdxl-turbo"}
ICON = {"queued": "⏳", "running": "🎬", "done": "✅", "cached": "♻️", "failed": "❌"}

st.set_page_config(page_title="ReelBatch", page_icon="🎞️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
.hero {padding: 2.4rem 2.6rem; border-radius: 20px; color: #fff; margin-bottom: 1.4rem;
       background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 55%, #06b6d4 100%);
       box-shadow: 0 10px 28px rgba(29, 78, 216, .28);}
.hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 2.4rem; letter-spacing: -.5px;}
.hero p {color: #dbeafe; margin: 0 0 1.1rem 0; font-size: 1.08rem; max-width: 72%;}
.chip {display: inline-block; padding: .28rem .85rem; margin: 0 .4rem .3rem 0; border-radius: 999px;
       background: rgba(255,255,255,.16); font-size: .85rem;}
.pill {display: inline-block; padding: .15rem .7rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
       background: rgba(29,78,216,.12); border: 1px solid rgba(29,78,216,.35); font-size: .85rem;}
.stButton > button {border-radius: 10px; font-weight: 600;}
.stButton > button[kind="primary"] {background: linear-gradient(120deg, #1d4ed8, #06b6d4); color: #fff; border: none; height: 3rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600; font-size: 1.02rem;}
</style>
<div class="hero">
  <h1>🎞️ ReelBatch</h1>
  <p>Drop in a spreadsheet of topics. Get a whole library of narrated, animated short videos, generated and saved automatically.</p>
  <span class="chip">📄 CSV / Excel in</span><span class="chip">🧠 AI scripts and visuals</span>
  <span class="chip">🎬 Motion</span><span class="chip">🎙️ Voiceover</span><span class="chip">📁 Videos out</span>
</div>
""", unsafe_allow_html=True)


def ingest(source, name):
    try:
        jobs, mapping = normalize(load_table(source, name))
        if not jobs:
            raise ValueError("No rows with a topic were found.")
        st.session_state.update(jobs=jobs, mapping=mapping, source=name)
    except Exception as e:  # shown to the user instead of a stack trace
        st.error(f"Could not read the sheet: {e}")


def build_runner():
    kind = ENGINES[st.session_state.get("engine", list(ENGINES)[0])]
    source = "openai" if kind == "demo" else IMAGE_SOURCES[st.session_state.get("image_source", list(IMAGE_SOURCES)[0])]
    model = VIDEO_MODELS[st.session_state.get("video_model", list(VIDEO_MODELS)[0])]
    problem = check_env(kind, source)
    if problem:
        return None, problem
    provider, motion = build_engines(kind, source, model)
    name = kind if kind == "demo" else f"{kind}-{source}"
    workers = effective_workers(kind, source, st.session_state.get("workers", 2))
    return BatchRunner(st.session_state["jobs"], provider, motion, LIB / name, BATCHES, workers,
                       st.session_state.get("reuse", True)), None


def monitor(runner):
    head, table = st.empty(), st.empty()

    def draw():
        rows = runner.rows
        finished = sum(runner.status[r["id"]]["state"] in ("done", "cached", "failed") for r in rows)
        head.progress(finished / len(rows), text=f"{finished} of {len(rows)} videos finished")
        df = pd.DataFrame([{"Topic": r["topic"],
                            "Status": f"{ICON[runner.status[r['id']]['state']]} {runner.status[r['id']]['stage']}",
                            "Progress": runner.status[r["id"]]["pct"] * 100} for r in rows])
        table.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "Progress": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")})

    while runner.running:
        draw()
        time.sleep(0.8)
    draw()
    states = [s["state"] for s in runner.status.values()]
    a, b, c, d = st.columns(4)
    a.metric("Created", states.count("done"))
    b.metric("Reused from cache", states.count("cached"))
    c.metric("Failed", states.count("failed"))
    d.metric("Time", f"{(runner.finished or time.time()) - runner.started:.0f}s")
    if sum(s["fallbacks"] for s in runner.status.values()):
        st.warning("Some local AI clips could not be generated (no GPU or a model error), "
                   "so the free camera-motion engine was used for those shots.")
    failed = {r["topic"]: runner.status[r["id"]]["error"] for r in runner.rows if runner.status[r["id"]]["state"] == "failed"}
    if failed:
        with st.expander("What went wrong", expanded=True):
            for topic, err in failed.items():
                st.error(f"**{topic}**: {err}")
    st.download_button("⬇️ Download manifest (CSV)", runner.manifest().to_csv(index=False), "manifest.csv", "text/csv")
    st.success("Done. Open the Library tab to watch and download your videos.")


tab_up, tab_run, tab_lib = st.tabs(["① Upload", "② Run", "③ Library"])

with tab_up:
    left, right = st.columns([2, 1], gap="large")
    upload = left.file_uploader("Your content sheet (CSV or Excel)", type=["csv", "xlsx", "xls"],
                                help="Only a 'topic' column is required. Everything else is optional.")
    right.write("")
    right.download_button("📄 Download template", Path("templates/sample.csv").read_bytes(), "reelbatch_template.csv", "text/csv",
                          use_container_width=True)
    if right.button("✨ Try the sample sheet", use_container_width=True):
        ingest("templates/sample.csv", "sample.csv")
    if upload and st.session_state.get("source") != upload.name:
        ingest(upload, upload.name)

    jobs = st.session_state.get("jobs")
    if jobs:
        found = [f"{key} ← {col}" for key, col in st.session_state["mapping"].items() if col]
        st.markdown("**Detected columns**  \n" + "".join(f"<span class='pill'>{f}</span>" for f in found), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(jobs).drop(columns=["id"]), hide_index=True, use_container_width=True)
        st.caption(f"{len(jobs)} videos ready (max {MAX_ROWS} per run). Missing columns use sensible defaults.")
    else:
        st.info("Upload a sheet or try the sample to begin. Columns: topic, audience, tone, style, duration, aspect, voice, "
                "language, character, motion, cta.")

    st.markdown("### Settings")
    s1, s2 = st.columns(2, gap="large")
    s1.radio("Motion engine", list(ENGINES), key="engine",
             help="Local AI video makes people and objects genuinely move but needs a GPU. The free engine animates the camera.")
    kind = ENGINES[st.session_state.get("engine", list(ENGINES)[0])]
    if kind == "local":
        s1.radio("Local video model", list(VIDEO_MODELS), key="video_model")
    if kind != "demo":
        s2.radio("Image source", list(IMAGE_SOURCES), key="image_source")
    s2.slider("Videos made in parallel (API-only runs)", 1, 4, 2, key="workers")
    s2.checkbox("Reuse videos already made for identical rows", True, key="reuse")
    st.caption(f"Local AI models: {gpu_status()[1]}")
    if jobs:
        n = sum(shots_for(j) for j in jobs)
        extra = f", plus **{n}** local AI video clips (minutes each on a consumer GPU)" if kind == "local" else ""
        st.info(f"This run will make about **{len(jobs)}** scripts and **{n}** voice clips and images{extra}.")

with tab_run:
    runner = st.session_state.get("runner")
    if not st.session_state.get("jobs") and not runner:
        st.info("Add a sheet in step ① first.")
    else:
        busy = bool(runner and runner.running)
        if st.button("🚀 Start batch", type="primary", use_container_width=True, disabled=busy):
            new_runner, problem = build_runner()
            if problem:
                st.warning(problem)
            else:
                new_runner.start()
                st.session_state["runner"] = runner = new_runner
        if runner:
            monitor(runner)

with tab_lib:
    videos = sorted(LIB.rglob("video.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not videos:
        st.info("Your finished videos will appear here.")
    else:
        top, zip_col = st.columns([3, 1])
        top.markdown(f"### {len(videos)} videos")
        if zip_col.button("📦 Prepare ZIP of all", use_container_width=True):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for v in videos:
                    z.write(v, f"{v.parent.name}.mp4")
            st.session_state["zip"] = buf.getvalue()
        if st.session_state.get("zip"):
            zip_col.download_button("⬇️ Download ZIP", st.session_state["zip"], "reelbatch_videos.zip", "application/zip",
                                    use_container_width=True)
        cols = st.columns(3, gap="medium")
        for i, v in enumerate(videos):
            plan_file = v.parent / "plan.json"
            title = json.loads(plan_file.read_text()).get("title", v.parent.name) if plan_file.exists() else v.parent.name
            with cols[i % 3].container(border=True):
                st.video(str(v))
                st.markdown(f"**{title}**")
                st.caption(f"{v.parent.parent.name} engine")
                st.download_button("Download", v.read_bytes(), f"{v.parent.name}.mp4", "video/mp4", key=f"dl_{v}",
                                   use_container_width=True)
