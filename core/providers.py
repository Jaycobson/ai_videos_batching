"""Providers: OpenAI (script, images, voice) and a no-API demo provider."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from .util import retry
from .video import ff

MODELS = [os.getenv("OPENAI_PLANNER_MODEL"), "gpt-5.6", "gpt-4.1", "gpt-4o-mini"]


def clean_plan(data: dict, row: dict, n: int) -> dict:
    shots = []
    for s in (data.get("shots") or [])[:n]:
        if str(s.get("image_prompt", "")).strip() and str(s.get("narration", "")).strip():
            shots.append({"image_prompt": str(s["image_prompt"]).strip(), "narration": str(s["narration"]).strip(),
                          "motion_prompt": str(s.get("motion_prompt") or row["motion"] or "natural gentle movement").strip()})
    if not shots:
        raise ValueError("The planner returned no usable shots.")
    return {"title": str(data.get("title") or row["topic"]).strip(),
            "character": str(data.get("character") or row["character"]).strip(), "shots": shots}


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def plan(self, row, n):
        system = (f"You are a short-video director. Reply with ONLY JSON: {{\"title\": str, \"character\": str, \"shots\": "
                  f"[{{\"image_prompt\": str, \"motion_prompt\": str, \"narration\": str}}]}} with exactly {n} shots.\n"
                  "- character: one fixed visual description of the recurring person (age, hair, clothing colours). "
                  "Use the one the user gives, else invent one.\n"
                  "- image_prompt: one vivid scene featuring that person doing something; no text in the image.\n"
                  "- motion_prompt: how the person and camera move in a 5-second clip (e.g. 'she walks toward the camera, "
                  "smiling, hair moving in the breeze').\n"
                  f"- narration: at most 12 words per shot, in {row['language']}, tone: {row['tone']}, for {row['audience']}. "
                  "The last shot ends with the call to action if one is given. Never invent facts or claims.")
        user = json.dumps({k: row[k] for k in ("topic", "character", "motion", "cta")})
        last = None
        for model in [m for m in MODELS if m]:
            try:
                resp = self.client.chat.completions.create(
                    model=model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return clean_plan(json.loads(resp.choices[0].message.content), row, n)
            except Exception as e:
                last = e
        raise RuntimeError(f"Planning failed: {last}")

    @retry()
    def image(self, prompt, size):
        res = self.client.images.generate(model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"), prompt=prompt,
                                          size=size, quality="low")
        item = res.data[0]
        return base64.b64decode(item.b64_json) if getattr(item, "b64_json", None) else requests.get(item.url, timeout=60).content

    @retry()
    def speak(self, text, out, voice, language):
        try:
            resp = self.client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text, response_format="mp3",
                                                   instructions=f"Narrate in {language}. Warm, natural and engaging.")
        except Exception:
            resp = self.client.audio.speech.create(model="tts-1", voice="nova", input=text, response_format="mp3")
        Path(out).write_bytes(resp.content)


class DemoProvider:
    """Runs the whole pipeline with no API calls: placeholder visuals and silent narration."""
    name = "demo"

    def plan(self, row, n):
        beats = ["Opening", "The moment", "The detail", "The result", "The takeaway", "Call to action"]
        shots = [{"image_prompt": f"{beats[i]}: {row['topic']}", "motion_prompt": row["motion"] or "gentle movement",
                  "narration": f"{beats[i]} of {row['topic']}."} for i in range(n)]
        return {"title": row["topic"], "character": row["character"], "shots": shots}

    def image(self, prompt, size):
        w, h = (int(v) for v in size.split("x"))
        seed = int(hashlib.sha1(prompt.encode()).hexdigest()[:6], 16)
        top, bottom = (seed % 160 + 40, seed // 7 % 160 + 40, seed // 31 % 160 + 40), (30, 30, 60)
        img = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(img)
        for y in range(h):
            t = y / h
            d.line((0, y, w, y), fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom, strict=True)))
        font = ImageFont.load_default(size=h // 28)
        d.multiline_text((w // 12, h // 3), "\n".join(prompt[i:i + 28] for i in range(0, len(prompt), 28)), font=font, fill="white")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    def speak(self, text, out, voice, language):
        ff("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", f"{max(1.2, len(text.split()) * 0.38):.2f}", out)
