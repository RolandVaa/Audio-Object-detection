#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized pipeline for RTX A6000 + Xeon Gold 6234
-------------------------------------------------
- Loads audio (.wav) under <root>/<round_dir>/<SITE>/<STATION>/<YYYYMMDD_%H%M%S>.wav
- Computes ONE mel-spectrogram for the full file, then slices into fixed-length segments
- Renders spectrograms WITHOUT matplotlib figures (uses magma colormap mapping)
- Batches Faster R-CNN inference on GPU (cuDNN benchmark enabled)
- Saves spectrogram PNGs ONLY when a segment has detections
- Writes one CSV row per detection
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import gc
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image

import librosa
import torch
import torchvision
import torchvision.transforms as T
from matplotlib import cm

# -----------------------------
# Exact audio/mel parameters
# -----------------------------
SEGMENT_DURATION_DEFAULT = 5
SR_TARGET_DEFAULT = 24000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 0.0
FMAX = 8000.0
POWER = 2.0
TOP_DB = None
IMG_RESIZE = (640, 640)  # (W, H) or None

# -----------------------------
# Torch & GPU config
# -----------------------------
def setup_torch(device: str, enable_benchmark: bool = True):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable_benchmark
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

# -----------------------------
# Path parsing utilities
# -----------------------------
def parse_site_station_from_path(path: Path, round_dir_name: str) -> Tuple[str, Optional[str]]:
    parts = path.parts
    try:
        i = parts.index(round_dir_name)
    except ValueError:
        i = None
        for idx, p in enumerate(parts):
            if p.lower() == round_dir_name.lower():
                i = idx
                break
        if i is None:
            return ("", None)
    site = parts[i+1] if i + 1 < len(parts) else ""
    station = parts[i+2] if i + 2 < len(parts) else None
    return site, station

def parse_timestamp_from_filename(stem: str) -> Optional[datetime]:
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return None

def iterate_wavs(root: Path, round_dir: str) -> Iterable[Path]:
    base = root / round_dir
    for p in base.rglob("*.wav"):
        if p.is_file():
            yield p

# -----------------------------
# Spectrogram generation
# -----------------------------
_MAGMA = cm.get_cmap("magma")

def power_to_db_per_segment(S_seg: np.ndarray) -> np.ndarray:
    maxv = S_seg.max()
    if maxv <= 0:
        return np.zeros_like(S_seg, dtype=np.float32)
    S_db = 10.0 * np.log10(np.maximum(S_seg, 1e-20) / maxv)
    return S_db.astype(np.float32)

def spectro_array_to_pil(S_db: np.ndarray) -> Image.Image:
    vmin = float(S_db.min())
    vmax = float(S_db.max())
    denom = max(vmax - vmin, 1e-12)
    x = (S_db - vmin) / denom
    x = np.flipud(x)
    rgba = _MAGMA(x)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if IMG_RESIZE is not None:
        w, h = IMG_RESIZE
        img = img.resize((w, h), resample=Image.BILINEAR)
    return img

def compute_full_mel(y: np.ndarray, sr: int) -> np.ndarray:
    S_full = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=POWER,
        center=True,
        window="hann",
        pad_mode="reflect",
    )
    return S_full.astype(np.float32, copy=False)

def frames_for_segment(sr: int, seg_seconds: int, hop_length: int) -> int:
    return int(seg_seconds * sr / hop_length)

# -----------------------------
# CSV Row struct
# -----------------------------
@dataclass
class DetectionRow:
    site: str
    station: Optional[str]
    timestamp: Optional[str]
    date: Optional[str]
    time: Optional[str]
    species: str
    confidence: float
    audio_path: str
    segment_index: int
    offset_seconds: float
    image_path: str

# -----------------------------
# Main runner
# -----------------------------
def run_pipeline(
    model_path: Path,
    root: Path,
    round_dir: str,
    csv_out: Path,
    class_names: List[str],
    device: str = "cuda:0",
    conf: float = 0.7,
    segment_duration: int = SEGMENT_DURATION_DEFAULT,
    sr_target: int = SR_TARGET_DEFAULT,
    batch: int = 4,
    img_out: Optional[Path] = None,
) -> None:
    setup_torch(device)

    # Load Faster R-CNN model
    num_classes = len(class_names)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = T.ToTensor()

    if img_out is None:
        img_out = csv_out.parent / "spectrosrcnn"
    img_out.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_out.exists()

    with open(csv_out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "site", "station", "timestamp", "date", "time",
                "species", "confidence", "audio_path", "segment_index", "offset_seconds", "image_path"
            ])

        wav_files = list(iterate_wavs(root, round_dir))
        for wav_path in tqdm(wav_files, desc="Processing WAV files", unit="file"):
            try:
                site, station = parse_site_station_from_path(wav_path, round_dir)
                base_dt = parse_timestamp_from_filename(wav_path.stem)

                y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True, dtype=np.float32, res_type="kaiser_fast")
                total_samples = len(y)
                samples_per_segment = segment_duration * sr
                if samples_per_segment <= 0 or total_samples < samples_per_segment:
                    del y
                    continue

                S_full = compute_full_mel(y, sr)
                del y

                frames_per_seg = frames_for_segment(sr, segment_duration, HOP_LENGTH)
                total_frames = S_full.shape[1]
                num_segments = total_samples // samples_per_segment

                seg_imgs: List[Image.Image] = []
                seg_meta: List[Tuple] = []

                for i in range(num_segments):
                    f0 = i * frames_per_seg
                    f1 = f0 + frames_per_seg
                    if f1 > total_frames:
                        break
                    S_seg = S_full[:, f0:f1]
                    S_db = power_to_db_per_segment(S_seg)
                    img = spectro_array_to_pil(S_db)
                    seg_imgs.append(img)

                    seg_offset_sec = i * segment_duration
                    if base_dt is not None:
                        seg_dt = base_dt + timedelta(seconds=seg_offset_sec)
                        ts_iso = seg_dt.strftime("%Y-%m-%d %H:%M:%S")
                        date_str = seg_dt.strftime("%Y-%m-%d")
                        time_str = seg_dt.strftime("%H:%M:%S")
                    else:
                        ts_iso = ""
                        date_str = ""
                        time_str = ""
                    seg_meta.append((
                        site, station, ts_iso, date_str, time_str,
                        i + 1, float(seg_offset_sec), str(wav_path.stem)
                    ))

                # Run in batches
                n = len(seg_imgs)
                start = 0
                while start < n:
                    end = min(start + batch, n)
                    batch_imgs = seg_imgs[start:end]
                    batch_meta = seg_meta[start:end]
                    inputs = [transform(img).to(device) for img in batch_imgs]

                    with torch.no_grad():
                        results = model(inputs)

                    for res, meta, pil_img in zip(results, batch_meta, batch_imgs):
                        site, station, ts_iso, date_str, time_str, seg_idx, seg_offset_sec, wavstem = meta
                        boxes = res["boxes"].cpu().numpy()
                        labels = res["labels"].cpu().numpy()
                        scores = res["scores"].cpu().numpy()
                        keep = scores >= conf

                        if keep.any():
                            out_dir = img_out / site / (station or "")
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / f"{wavstem}_seg{seg_idx}.png"
                            pil_img.save(str(out_path))
                            img_path_str = str(out_path)

                            for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                                species = class_names[label] if label < len(class_names) else str(label)
                                writer.writerow([
                                    site, station or "", ts_iso, date_str, time_str,
                                    species, f"{score:.4f}", str(wav_path),
                                    int(seg_idx), float(seg_offset_sec), img_path_str
                                ])

                    del results, inputs, batch_imgs, batch_meta
                    gc.collect()
                    start = end

                del S_full, seg_imgs, seg_meta
                gc.collect()

            except Exception as e:
                sys.stderr.write(f"[WARN] Failed on {wav_path}: {e}\n")
                continue

def main():
    parser = argparse.ArgumentParser(description="Fast Faster R-CNN inference on mel-spectrogram segments.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--round_dir", type=str, default="1_round")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--csv_out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--segment_duration", type=int, default=SEGMENT_DURATION_DEFAULT)
    parser.add_argument("--sr_target", type=int, default=SR_TARGET_DEFAULT)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--img_out", type=str, default=None)
    parser.add_argument("--classes", type=str, required=True, help="Comma-separated list of class names")

    #args = parser.parse_args()


    args = parser.parse_args([
        "--root", r"F:\BirdData2025",
        "--round_dir", "SpeedTest",  # Change to "2_round" for the other dataset
        "--model", r"C:\Users\admin\Desktop\thesisalgos\fasterrcnn_owlv6_best.pth",
        "--csv_out", r"F:\BirdData2025\SpeedTest.csv",
        "--device", "cuda:0",
        "--conf", "0.7",
        "--segment_duration", str(SEGMENT_DURATION_DEFAULT),
        "--sr_target", str(SR_TARGET_DEFAULT),
        "--batch", "4",
        "--classes", "__background__,Aegolius Funereus,Glaucidium passerinum,Strix Uralensis"
        # replace with your real class list
    ])

    class_names = args.classes.split(",")
    img_out = Path(args.img_out) if args.img_out is not None else None

    run_pipeline(
        model_path=Path(args.model),
        root=Path(args.root),
        round_dir=args.round_dir,
        csv_out=Path(args.csv_out),
        class_names=class_names,
        device=args.device,
        conf=args.conf,
        segment_duration=args.segment_duration,
        sr_target=args.sr_target,
        batch=args.batch,
        img_out=img_out,
    )

if __name__ == "__main__":
    main()
