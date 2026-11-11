"""FFmpeg helpers: living-image motion, fitting AI clips, and final assembly."""
import shutil
import subprocess
import tempfile
from pathlib import Path


def ffmpeg_ok() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def duration(path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                          str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# Camera moves for the free "living image" engine (n = number of frames).
MOVES = [
    "z='min(1.0+0.0016*on,1.28)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "z='max(1.28-0.0016*on,1.0)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "z=1.22:x='(iw-iw/zoom)*on/{n}':y='ih/2-(ih/zoom/2)'",
    "z=1.22:x='(iw-iw/zoom)*(1-on/{n})':y='ih/2-(ih/zoom/2)'",
    "z=1.22:x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{n})'",
]


def living_clip(image, out, dur, size, variant=0, fps=30):
    """Free motion: a graded, vignetted camera move over the image."""
    w, h = size
    frames = int(dur * fps)
    move = MOVES[variant % len(MOVES)].format(n=frames)
    vf = (f"scale={2 * w}:{2 * h}:force_original_aspect_ratio=increase,crop={2 * w}:{2 * h},"
          f"zoompan={move}:d={frames}:s={w}x{h}:fps={fps},eq=saturation=1.12:contrast=1.04,vignette=PI/6,format=yuv420p")
    ff("-i", image, "-vf", vf, "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-r", fps, out)


def fit_clip(src, out, dur, size, fps=30):
    """Fit an AI-generated clip to the target size and exact duration (freezes the last frame if it is too short)."""
    w, h = size
    clip = duration(src)
    stretch = min(max(dur / clip, 1.0), 1.6) if clip else 1.0  # gentle slow-motion, then freeze any remainder
    vf = (f"setpts={stretch:.3f}*PTS,scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},"
          f"tpad=stop_mode=clone:stop_duration={dur:.2f},format=yuv420p")
    ff("-i", src, "-vf", vf, "-t", f"{dur:.2f}", "-an", "-c:v", "libx264", "-preset", "veryfast", out)


def assemble(clips, voices, out_path, durs):
    with tempfile.TemporaryDirectory() as tmp:
        segs = []
        for i, (clip, voice, d) in enumerate(zip(clips, voices, durs, strict=True)):
            seg = Path(tmp) / f"seg_{i}.mp4"
            ff("-i", clip, "-i", voice, "-map", "0:v", "-map", "1:a", "-af", "apad", "-t", f"{d:.2f}", "-c:v", "libx264",
               "-preset", "veryfast", "-c:a", "aac", "-ar", 44100, "-ac", 2, seg)
            segs.append(seg)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs))
        ff("-f", "concat", "-safe", 0, "-i", listing, "-c", "copy", out_path)


def _ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(texts, durs, path):
    t, out = 0.0, []
    for i, (text, d) in enumerate(zip(texts, durs, strict=True), 1):
        out.append(f"{i}\n{_ts(t)} --> {_ts(t + d)}\n{text}\n")
        t += d
    Path(path).write_text("\n".join(out), encoding="utf-8")
