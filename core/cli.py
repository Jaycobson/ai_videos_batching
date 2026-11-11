"""Headless runner (ideal for Colab): python -m core.cli sheet.csv --engine local --video-model ltx"""
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .batch import BatchRunner
from .engines import build_engines, check_env, effective_workers
from .sheet import load_table, normalize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate a library of videos from a CSV/Excel sheet.")
    ap.add_argument("sheet")
    ap.add_argument("--engine", choices=["living", "local", "demo"], default="living")
    ap.add_argument("--video-model", choices=["ltx", "svd"], default="ltx")
    ap.add_argument("--images", choices=["openai", "sdxl-turbo"], default="openai")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    load_dotenv()
    problem = check_env(args.engine, args.images)
    if problem:
        print(f"Cannot start: {problem}")
        return 2
    rows, _ = normalize(load_table(args.sheet))
    provider, motion = build_engines(args.engine, args.images, args.video_model)
    name = args.engine if args.engine == "demo" else f"{args.engine}-{args.images}"
    runner = BatchRunner(rows, provider, motion, Path(args.out) / "library" / name, Path(args.out) / "batches",
                         effective_workers(args.engine, args.images, args.workers), not args.no_cache)
    runner.start()
    seen = {}
    while runner.running:
        for r in rows:
            s = runner.status[r["id"]]
            if seen.get(r["id"]) != (s["state"], s["stage"]):
                seen[r["id"]] = (s["state"], s["stage"])
                print(f"[{s['state']:>7}] {r['topic'][:50]:<50} {s['stage']}")
        time.sleep(2)
    states = [s["state"] for s in runner.status.values()]
    print(f"\nDone: {states.count('done')} created, {states.count('cached')} cached, {states.count('failed')} failed")
    print(f"Manifest: {runner.manifest_path}")
    return 1 if "failed" in states else 0


if __name__ == "__main__":
    sys.exit(main())
