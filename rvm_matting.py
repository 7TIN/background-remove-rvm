#!/usr/bin/env python3
"""
RVM ResNet50 Video Matting — Production Script
Converts: Video → VP9 WebM with alpha (yuva420p)

Usage:
    python rvm_matting.py <input_video> <output_webm>

Or import and call:
    from rvm_matting import process_video
    output_path = process_video("input.mp4", "output.webm")
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import shutil

# ── Configuration ──────────────────────────────────────────────────────────
RVM_REPO_URL = "https://github.com/PeterL1n/RobustVideoMatting.git"
RVM_DIR = Path(__file__).parent / "RobustVideoMatting"
MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"
MODEL_PATH = Path(__file__).parent / "rvm_resnet50.pth"

# Auto-detect CUDA
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ── Singleton model cache ──────────────────────────────────────────────────
_model = None

def ensure_rvm_repo():
    """Clone RVM repo if not present."""
    if RVM_DIR.exists():
        print(f"[RVM] Repo already exists at {RVM_DIR}")
        return

    print(f"[RVM] Cloning repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", RVM_REPO_URL, str(RVM_DIR)],
        check=True,
        capture_output=True,
    )
    print(f"[RVM] Repo cloned.")

def ensure_model():
    """Download ResNet50 checkpoint if not present."""
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1e6
        print(f"[RVM] Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        return

    print(f"[RVM] Downloading model...")
    subprocess.run(
        ["curl", "-L", "-o", str(MODEL_PATH), MODEL_URL],
        check=True,
        capture_output=True,
    )
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"[RVM] Model downloaded: {size_mb:.1f} MB")

def patch_inference_utils():
    """Overwrite inference_utils.py with PyAV 12.x compatible version."""
    patch = """import av
import torch
import numpy as np
from torch.utils.data import Dataset

class VideoReader(Dataset):
    def __init__(self, path, transform=None):
        self.frames = []
        self.transform = transform
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for frame in container.decode(stream):
            self.frames.append(frame.to_ndarray(format='rgb24'))
        container.close()
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        t = torch.from_numpy(self.frames[idx]).permute(2,0,1).float()/255.0
        return self.transform(t) if self.transform else t

class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=None):
        self.container = av.open(path, mode='w')
        self.stream = self.container.add_stream('h264', rate=int(frame_rate))
        if bit_rate: self.stream.bit_rate = int(bit_rate)
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '18'}
    def write(self, frames):
        if frames.ndim == 3: frames = frames.unsqueeze(0)
        frames = (frames.clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
        for f in frames:
            pkt = self.stream.encode(av.VideoFrame.from_ndarray(f[...,:3], format='rgb24'))
            for p in pkt: self.container.mux(p)
    def close(self):
        for p in self.stream.encode(): self.container.mux(p)
        self.container.close()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
"""
    target = RVM_DIR / "inference_utils.py"
    target.write_text(patch)
    print("[RVM] inference_utils.py patched for PyAV 12.x")

def setup():
    """One-time setup: clone repo, download model, patch utils."""
    ensure_rvm_repo()
    ensure_model()
    patch_inference_utils()

    # Add RVM to path for imports
    if str(RVM_DIR) not in sys.path:
        sys.path.insert(0, str(RVM_DIR))

def load_model():
    """Lazy-load model (singleton)."""
    global _model
    if _model is not None:
        return _model

    import torch
    from model import MattingNetwork

    _model = MattingNetwork(variant="resnet50").eval().to(device)
    _model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    print(f"[RVM] Model loaded on {device}")
    return _model

def get_video_info(path: str) -> Tuple[int, int, float, int]:
    """Returns (width, height, fps, total_frames)."""
    import av
    c = av.open(path)
    s = c.streams.video[0]
    w = s.codec_context.width
    h = s.codec_context.height
    fps = float(s.average_rate)
    total = s.frames
    c.close()
    return w, h, fps, total

def auto_downsample_ratio(width: int, height: int, target_short: int = 384) -> float:
    """Auto-suggest downsample ratio so short side becomes ~384px."""
    shorter = min(width, height)
    ratio = round(target_short / shorter, 3)
    return max(0.1, min(1.0, ratio))

def process_video(
    input_path: str,
    output_path: str,
    downsample_ratio: Optional[float] = None,
    seq_chunk: int = 12,
    crf: int = 30,
    bitrate: Optional[str] = None,
) -> str:
    """
    Process a video through RVM and output VP9 WebM with alpha (yuva420p).

    Chrome/Remotion compatible — no ffmpeg post-processing needed.

    Args:
        input_path: Path to input video (any format ffmpeg can read)
        output_path: Path for output .webm (VP9 with alpha)
        downsample_ratio: Override auto-detected ratio (None = auto)
        seq_chunk: Frames per batch (lower = better quality, slower)
        crf: VP9 quality (0-63, lower = better, default 30 for speed/quality balance)
        bitrate: Optional bitrate override (e.g. "2M", "5M")

    Returns:
        Absolute path to output file
    """
    import torch
    import av
    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from inference_utils import VideoReader
    from fractions import Fraction

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get video info
    W, H, FPS, TOTAL_FRAMES = get_video_info(str(input_path))
    print(f"[RVM] Input: {W}x{H} @ {FPS:.3f}fps, {TOTAL_FRAMES} frames")

    # Auto ratio
    if downsample_ratio is None:
        downsample_ratio = auto_downsample_ratio(W, H)
    print(f"[RVM] downsample_ratio={downsample_ratio}, seq_chunk={seq_chunk}")

    # Load model
    model = load_model()

    # Setup VP9 WebM writer with alpha
    out_webm = av.open(str(output_path), mode="w", format="webm")
    vs = out_webm.add_stream("libvpx-vp9", rate=Fraction(FPS).limit_denominator())
    vs.width = W
    vs.height = H
    vs.pix_fmt = "yuva420p"           # Chrome-compatible alpha

    # VP9 options for alpha + quality
    options = {
        "auto-alt-ref": "0",          # REQUIRED for alpha channel in VP9
        "crf": str(crf),
        "b:v": bitrate or "0",        # 0 = VBR with CRF
        "row-mt": "1",                # Multi-threading
        "cpu-used": "4",              # Speed/quality tradeoff (0-8, higher = faster)
        "deadline": "good",           # Encoding quality target
    }
    if bitrate:
        options["b:v"] = bitrate
        options["crf"] = "0"          # Disable CRF when bitrate is set

    vs.options = options

    # Process
    reader = VideoReader(str(input_path), transform=None)
    loader = DataLoader(reader, batch_size=seq_chunk, shuffle=False)

    rec = [None] * 4
    frame_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, unit="chunk", desc="Matting"):
            src = batch.to(device).unsqueeze(0)
            fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)

            fgr = fgr.squeeze(0)
            pha = pha.squeeze(0)

            for i in range(fgr.shape[0]):
                rgb_u8 = (fgr[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                alpha_u8 = (pha[i].clamp(0, 1) * 255).byte().squeeze(0).cpu().numpy()

                # Build yuva420p frame directly
                rgba_u8 = np.concatenate([rgb_u8, alpha_u8[..., None]], axis=2)
                av_rgba = av.VideoFrame.from_ndarray(rgba_u8, format="rgba")
                av_yuva = av_rgba.reformat(format="yuva420p")

                for pkt in vs.encode(av_yuva):
                    out_webm.mux(pkt)

                frame_count += 1

    # Flush
    for pkt in vs.encode():
        out_webm.mux(pkt)
    out_webm.close()

    size_mb = output_path.stat().st_size / 1e6
    print(f"[RVM] Done — {frame_count} frames → {output_path} ({size_mb:.1f} MB)")
    return str(output_path)

def main() -> int:
    """CLI entrypoint."""
    if len(sys.argv) < 3:
        print("Usage: python rvm_matting.py <input_video> <output_webm>")
        return 1

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Setup (one-time)
    setup()

    # Process
    process_video(input_file, output_file)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())