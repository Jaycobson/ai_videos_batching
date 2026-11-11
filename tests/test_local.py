import sys
import types

import pytest
from PIL import Image

from core import local_models
from core.cli import main
from core.video import ff, ffmpeg_ok


def png_bytes():
    import io
    buf = io.BytesIO()
    Image.new("RGB", (64, 96), (120, 80, 200)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def fake_stack(monkeypatch):
    """A fake torch + diffusers so the local-model code paths run without a GPU."""
    state = {"cuda": True, "calls": [], "loads": []}
    torch = types.ModuleType("torch")
    torch.float16, torch.bfloat16 = "fp16", "bf16"
    torch.Generator = lambda device="cpu": types.SimpleNamespace(manual_seed=lambda s: s)
    torch.cuda = types.SimpleNamespace(is_available=lambda: state["cuda"], empty_cache=lambda: None,
                                       get_device_name=lambda i: "Fake GPU")
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    class Pipe:
        vae = types.SimpleNamespace(enable_tiling=lambda: None)

        @classmethod
        def from_pretrained(cls, model_id, **kw):
            state["loads"].append((model_id, kw))
            return cls()

        def enable_model_cpu_offload(self):
            state["offload"] = True

        def __call__(self, *a, **kw):
            state["calls"].append(kw)
            return types.SimpleNamespace(frames=[[Image.new("RGB", (32, 32))] * 3], images=[Image.new("RGB", (64, 64))])

    diffusers = types.ModuleType("diffusers")
    diffusers.LTXImageToVideoPipeline = diffusers.StableVideoDiffusionPipeline = diffusers.AutoPipelineForText2Image = Pipe
    utils = types.ModuleType("diffusers.utils")
    utils.export_to_video = lambda frames, path, fps=24: ff("-f", "lavfi", "-i", "testsrc=duration=2:size=128x128:rate=24", path)
    for name, mod in (("torch", torch), ("diffusers", diffusers), ("diffusers.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    return state


@pytest.mark.skipif(not ffmpeg_ok(), reason="FFmpeg not installed")
def test_ltx_uses_valid_dimensions_and_frame_count(fake_stack, tmp_path):
    video = local_models.LocalVideo("ltx")
    video.animate(png_bytes(), "she walks toward the camera", tmp_path / "a.mp4", size=(720, 1280))
    kw = fake_stack["calls"][0]
    assert kw["width"] % 32 == 0 and kw["height"] % 32 == 0 and kw["height"] > kw["width"]
    assert (kw["num_frames"] - 1) % 8 == 0 and fake_stack["offload"] and (tmp_path / "a.mp4").exists()


@pytest.mark.skipif(not ffmpeg_ok(), reason="FFmpeg not installed")
def test_svd_loads_fp16_and_chunks_decoding(fake_stack, tmp_path):
    local_models.LocalVideo("svd").animate(png_bytes(), "ignored", tmp_path / "b.mp4", size=(1280, 720))
    assert fake_stack["loads"][0][1]["variant"] == "fp16"
    assert fake_stack["calls"][0]["decode_chunk_size"] == 4 and fake_stack["calls"][0]["num_frames"] == 25


def test_no_gpu_is_refused_clearly_and_remembered(fake_stack, tmp_path, monkeypatch):
    fake_stack["cuda"] = False
    monkeypatch.delenv("ALLOW_CPU", raising=False)
    video = local_models.LocalVideo("ltx")
    for _ in range(2):  # the second call fails instantly from the remembered error
        with pytest.raises(RuntimeError, match="No GPU"):
            video.animate(png_bytes(), "x", tmp_path / "c.mp4")
    assert len(fake_stack["loads"]) == 0


def test_local_images_return_png(fake_stack):
    data = local_models.LocalImages().image("a cat", "1024x1536")
    assert data[:4] == b"\x89PNG" and fake_stack["calls"][0]["num_inference_steps"] == 2


def test_gpu_status_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert local_models.gpu_status()[0] == "none"


@pytest.mark.skipif(not ffmpeg_ok(), reason="FFmpeg not installed")
def test_cli_runs_in_demo_mode(tmp_path):
    sheet = tmp_path / "s.csv"
    sheet.write_text("topic,duration,aspect\nCLI demo,10,1:1\n")
    assert main([str(sheet), "--engine", "demo", "--out", str(tmp_path / "out")]) == 0
    assert list((tmp_path / "out").rglob("video.mp4"))
