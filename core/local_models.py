"""Open-source local models via Hugging Face diffusers: SDXL-Turbo (images), SVD-XT and LTX-Video (image-to-video).

Everything is imported lazily so the app runs without PyTorch. These classes need a GPU (see requirements-gpu.txt).
"""
import hashlib
import io
import os
import threading

from PIL import Image

# OpenAI-style size string -> (width, height) that local models handle well (multiples of 32/64)
IMAGE_SIZES = {"1024x1536": (576, 1024), "1024x1024": (512, 512), "1536x1024": (1024, 576)}
NEGATIVE = "worst quality, blurry, jittery, inconsistent motion, distorted face, extra limbs"
_gpu_lock = threading.Lock()  # one generation at a time per GPU


def pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "none"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and mps.is_available() else "cpu"


def gpu_status() -> tuple[str, str]:
    device = pick_device()
    if device == "none":
        return device, "⚪ PyTorch not installed (optional: pip install -r requirements-gpu.txt)"
    if device == "cuda":
        import torch
        return device, f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}"
    if device == "mps":
        return device, "✅ Apple Silicon GPU detected"
    return device, "⚪ No GPU detected here. Run local models on Colab (see notebooks/) or fall back to the free engine"


def _require_gpu() -> str:
    device = pick_device()
    if device == "none":
        raise RuntimeError("PyTorch/diffusers are not installed. Run: pip install -r requirements-gpu.txt")
    if device == "cpu" and not os.getenv("ALLOW_CPU"):
        raise RuntimeError("No GPU found. Local diffusion models need a CUDA or Apple GPU. Use the Colab notebook, "
                           "or set ALLOW_CPU=1 to force CPU (extremely slow).")
    return device


def _place(pipe, device):
    """On CUDA, offload idle sub-models to CPU RAM so 12-16 GB cards (and Colab T4s) can fit the pipeline."""
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def _generator(text):
    import torch
    return torch.Generator(device="cpu").manual_seed(int(hashlib.sha1(text.encode()).hexdigest()[:8], 16))


def _free_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LocalImages:
    """Text-to-image with SDXL-Turbo: 2 steps, no classifier-free guidance, seeded per prompt for repeatability."""

    MODEL = "stabilityai/sdxl-turbo"

    def __init__(self):
        self.pipe = None

    def _load(self):
        import torch
        from diffusers import AutoPipelineForText2Image
        device = _require_gpu()
        return _place(AutoPipelineForText2Image.from_pretrained(self.MODEL, torch_dtype=torch.float16, variant="fp16"), device)

    def image(self, prompt, size):
        w, h = IMAGE_SIZES.get(size, (512, 512))
        with _gpu_lock:
            self.pipe = self.pipe or self._load()
            img = self.pipe(prompt=prompt, width=w, height=h, num_inference_steps=2, guidance_scale=0.0,
                            generator=_generator(prompt)).images[0]
            _free_memory()
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()


class LocalVideo:
    """Image-to-video on your own GPU.

    ltx: Lightricks LTX-Video. Follows the text motion prompt. 97 frames (8k+1) at 24 fps, sizes divisible by 32.
    svd: Stable Video Diffusion XT. Ignores text, animates the image by itself. 25 frames at 7 fps.
    """

    MODELS = {"svd": "stabilityai/stable-video-diffusion-img2vid-xt", "ltx": "Lightricks/LTX-Video"}

    def __init__(self, model="ltx"):
        if model not in self.MODELS:
            raise ValueError(f"Unknown video model: {model}")
        self.model, self.pipe, self.error = model, None, None

    def _load(self):
        import torch
        device = _require_gpu()
        if self.model == "svd":
            from diffusers import StableVideoDiffusionPipeline
            pipe = StableVideoDiffusionPipeline.from_pretrained(self.MODELS["svd"], torch_dtype=torch.float16, variant="fp16")
        else:
            from diffusers import LTXImageToVideoPipeline
            pipe = LTXImageToVideoPipeline.from_pretrained(self.MODELS["ltx"], torch_dtype=torch.bfloat16)
            pipe.vae.enable_tiling()  # decode the video in tiles to save VRAM
        return _place(pipe, device)

    def animate(self, image_bytes, prompt, out, size=None):
        from diffusers.utils import export_to_video
        portrait = bool(size and size[1] > size[0])
        square = bool(size and size[0] == size[1])
        with _gpu_lock:
            if self.error:
                raise self.error
            try:
                self.pipe = self.pipe or self._load()
            except Exception as e:  # remember the failure so every remaining shot falls back instantly
                self.error = e
                raise
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            gen = _generator(prompt)
            if self.model == "ltx":
                w, h = (512, 512) if square else (480, 704) if portrait else (704, 480)
                frames = self.pipe(image=image.resize((w, h)), prompt=f"{prompt}. Smooth natural motion, consistent identity",
                                   negative_prompt=NEGATIVE, width=w, height=h, num_frames=97, num_inference_steps=40,
                                   generator=gen).frames[0]
                fps = 24
            else:
                w, h = (576, 1024) if portrait else (1024, 576)
                frames = self.pipe(image.resize((w, h)), width=w, height=h, num_frames=25, decode_chunk_size=4,
                                   motion_bucket_id=127, noise_aug_strength=0.02, generator=gen).frames[0]
                fps = 7
            export_to_video(frames, str(out), fps=fps)
            _free_memory()
