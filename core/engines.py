"""Builds the provider and motion engine for a run (shared by the Streamlit app and the CLI)."""
import os

from .local_models import LocalImages, LocalVideo, gpu_status, pick_device
from .providers import DemoProvider, OpenAIProvider
from .video import ffmpeg_ok

__all__ = ["build_engines", "check_env", "effective_workers", "gpu_status"]


class ComposedProvider:
    """OpenAI plans and speaks; a different component draws the images."""
    name = "composed"

    def __init__(self, base, images):
        self.base, self.images = base, images

    def plan(self, row, n):
        return self.base.plan(row, n)

    def speak(self, *args):
        return self.base.speak(*args)

    def image(self, prompt, size):
        return self.images.image(prompt, size)


def check_env(kind, image_source="openai"):
    """Returns a human-readable problem, or None when the run can start."""
    if not ffmpeg_ok():
        return "FFmpeg/ffprobe not found. Install FFmpeg and make sure it is on your PATH."
    if kind != "demo" and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not found. Add it to your .env file, or pick Demo mode."
    if kind != "demo" and image_source == "sdxl-turbo" and pick_device() in ("none", "cpu") and not os.getenv("ALLOW_CPU"):
        return "Local SDXL-Turbo images need a GPU. Switch the image source to OpenAI, or run on Colab (see notebooks/)."
    return None


def build_engines(kind, image_source="openai", video_model="ltx"):
    base = DemoProvider() if kind == "demo" else OpenAIProvider(os.getenv("OPENAI_API_KEY"))
    provider = ComposedProvider(base, LocalImages()) if kind != "demo" and image_source == "sdxl-turbo" else base
    motion = LocalVideo(video_model) if kind == "local" else None
    return provider, motion


def effective_workers(kind, image_source, requested):
    """Local GPU models run one job at a time; API-only runs can go parallel."""
    return 1 if kind == "local" or image_source == "sdxl-turbo" else requested
