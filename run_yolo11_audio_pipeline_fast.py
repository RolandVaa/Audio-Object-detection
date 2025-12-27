#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized pipeline for RTX A6000 + Xeon Gold 6234
-------------------------------------------------
- Loads audio (.wav) under <root>/<round_dir>/<SITE>/<STATION>/<YYYYMMDD_%H%M%S>.wav
- Computes ONE mel-spectrogram for the full file, then slices into fixed-length segments
- Renders spectrograms WITHOUT matplotlib figures (uses magma colormap mapping)
- Batches YOLOv11 inference on GPU (FP16 by default), cuDNN benchmark enabled
- Saves spectrogram PNGs ONLY when a segment has detections
- Writes one CSV row per detection

CSV columns:
  site, station, timestamp, date, time, species, confidence, audio_path, segment_index, offset_seconds, image_path
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
import librosa.display  # keep for consistent dependency (not used for plotting)

import torch
from ultralytics import YOLO

# For colormap mapping identical to specshow(cmap='magma')
from matplotlib import cm

# -----------------------------
# Exact audio/mel parameters (match your original)
# -----------------------------
SEGMENT_DURATION_DEFAULT = 5          # seconds
SR_TARGET_DEFAULT = 24000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 0.0
FMAX = 8000.0
POWER = 2.0     # power for melspectrogram
TOP_DB = None   # None, because we use ref=np.max exactly (no top_db clipping in your code)

# Output "image" size: specshow used figsize=(5,2.5), dpi=100 -> ~500x250 px.
# We will *not* enforce exact pixel count; YOLO will resize to imgsz internally.
# If you want to force a specific size before YOLO, you can set IMG_RESIZE=(500, 250).
IMG_RESIZE = (640, 640)  # e.g., (500, 250) as (W, H); or None to leave native


# -----------------------------
# Torch & GPU config
# -----------------------------
def setup_torch(device: str, enable_benchmark: bool = True):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable_benchmark
        # Optional for Ampere+: can slightly help matmul throughput
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
# Spectrogram generation (fast, figure-free, same look)
# -----------------------------
_MAGMA = cm.get_cmap("magma")  # cached once

def power_to_db_per_segment(S_seg: np.ndarray) -> np.ndarray:
    """
    Match librosa.power_to_db(S, ref=np.max) per *segment*.
    This is equivalent to: 10*log10(S_seg / S_seg.max())
    """
    # Avoid division by zero
    maxv = S_seg.max()
    if maxv <= 0:
        # return constant -inf; but for visuals, just return all zeros
        return np.zeros_like(S_seg, dtype=np.float32)
    S_db = 10.0 * np.log10(np.maximum(S_seg, 1e-20) / maxv)
    return S_db.astype(np.float32)


def spectro_array_to_pil(S_db: np.ndarray) -> Image.Image:
    """
    Apply magma colormap as specshow would, with origin='lower'.
    """
    # Normalize per image (min..max => 0..1), like imshow default scaling
    vmin = float(S_db.min())
    vmax = float(S_db.max())
    denom = max(vmax - vmin, 1e-12)
    x = (S_db - vmin) / denom

    # specshow uses origin='lower' => flip vertically to match that orientation
    x = np.flipud(x)

    rgba = _MAGMA(x)  # returns float in [0,1], shape (H,W,4)
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
    # Keep as power; convert to dB per segment to preserve your original per-segment ref=np.max behavior
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
    device: str = "cuda:0",
    imgsz: int = 640,
    conf: float = 0.7,
    iou: float = 0.45,
    segment_duration: int = SEGMENT_DURATION_DEFAULT,
    sr_target: int = SR_TARGET_DEFAULT,
    batch: int = 16,
    half: bool = True,
    img_out: Optional[Path] = None,
) -> None:
    setup_torch(device)

    model = YOLO(str(model_path))
    if device:
        model.to(device)

    # Default image output directory
    if img_out is None:
        img_out = csv_out.parent / "spectrosyolo11Final"
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

        # Walk files
        wav_files = list(iterate_wavs(root,round_dir))
        for wav_path in tqdm(wav_files,desc="Processing WAV files",unit="file"):
        #for wav_path in iterate_wavs(root, round_dir):
            try:
                site, station = parse_site_station_from_path(wav_path, round_dir)
                base_dt = parse_timestamp_from_filename(wav_path.stem)

                # Load audio (libsndfile via librosa.load if installed)
                y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True, dtype=np.float32, res_type="kaiser_fast")
                total_samples = len(y)
                samples_per_segment = segment_duration * sr
                if samples_per_segment <= 0 or total_samples < samples_per_segment:
                    del y
                    continue

                # One mel for the whole file (power scale)
                S_full = compute_full_mel(y, sr)
                del y  # free audio

                frames_per_seg = frames_for_segment(sr, segment_duration, HOP_LENGTH)
                total_frames = S_full.shape[1]
                # Segment count based on samples for consistency with your original logic
                num_segments = total_samples // samples_per_segment

                # Prepare all PIL images for this file (fast path)
                # We'll keep them in memory briefly; with 200 GB RAM this is fine.
                seg_imgs: List[Image.Image] = []
                seg_meta: List[Tuple[str, Optional[str], str, str, str, int, float, str]] = []  # metadata per segment

                for i in range(num_segments):
                    # Approximate frame start from segment index
                    f0 = i * frames_per_seg
                    f1 = f0 + frames_per_seg
                    if f1 > total_frames:
                        break  # safety

                    S_seg = S_full[:, f0:f1]
                    S_db = power_to_db_per_segment(S_seg)
                    img = spectro_array_to_pil(S_db)
                    seg_imgs.append(img)

                    # Segment timestamp and meta
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
                        site,
                        station,
                        ts_iso,
                        date_str,
                        time_str,
                        i + 1,                # 1-based index
                        float(seg_offset_sec),
                        str(wav_path.stem)     # for image filename
                    ))

                # Now batch inference over seg_imgs
                # Process in chunks to control VRAM
                def run_batches(images: List[Image.Image], metas: List[Tuple], batch_size: int):
                    n = len(images)
                    start = 0
                    while start < n:
                        end = min(start + batch_size, n)
                        batch_imgs = images[start:end]
                        batch_meta = metas[start:end]

                        results = model.predict(
                            source=batch_imgs,
                            imgsz=imgsz,
                            conf=conf,
                            iou=iou,
                            device=device,
                            verbose=False,
                            half=half,
                            batch=batch_size
                        )

                        # For each result in the batch
                        for res, meta, pil_img in zip(results, batch_meta, batch_imgs):
                            class_names = res.names if hasattr(res, "names") else model.names
                            site, station, ts_iso, date_str, time_str, seg_idx, seg_offset_sec, wavstem = meta

                            if res.boxes is not None and len(res.boxes) > 0:
                                # Save the image because there is at least one detection
                                out_dir = img_out / site / (station or "")
                                out_dir.mkdir(parents=True, exist_ok=True)
                                out_path = out_dir / f"{wavstem}_seg{seg_idx}.png"
                                pil_img.save(str(out_path))
                                img_path_str = str(out_path)

                                # Write one row per detection
                                for b in res.boxes:
                                    cls_id = int(b.cls.item())
                                    conf_score = float(b.conf.item()) if b.conf is not None else float("nan")
                                    species = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)

                                    writer.writerow([
                                        site,
                                        station if station is not None else "",
                                        ts_iso,
                                        date_str,
                                        time_str,
                                        species,
                                        f"{conf_score:.4f}",
                                        str(wav_path),
                                        int(seg_idx),
                                        float(seg_offset_sec),
                                        img_path_str
                                    ])
                            # else: no detections -> no save, no row

                        # Cleanup batch
                        del results, batch_imgs, batch_meta
                        gc.collect()

                        start = end

                run_batches(seg_imgs, seg_meta, batch)

                # Cleanup per file
                del S_full, seg_imgs, seg_meta
                gc.collect()

            except Exception as e:
                sys.stderr.write(f"[WARN] Failed on {wav_path}: {e}\n")
                continue


def main():
    parser = argparse.ArgumentParser(description="Fast YOLOv11 inference on mel-spectrogram segments (save images on detection, batch GPU).")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing round_dir (e.g., /data/project)")
    parser.add_argument("--round_dir", type=str, default="1_round", help="Top-level folder with sites (default: 1_round)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Ultralytics YOLOv11 .pt model")
    parser.add_argument("--csv_out", type=str, required=True, help="Path to output CSV (created/appended)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (set to what you trained with)")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--segment_duration", type=int, default=SEGMENT_DURATION_DEFAULT, help="Segment length in seconds (default: 5)")
    parser.add_argument("--sr_target", type=int, default=SR_TARGET_DEFAULT, help="Target sampling rate for librosa.load (default: 24000)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for YOLO inference")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16 for GPU inference (default: True for CUDA)")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable FP16 inference")
    parser.add_argument("--img_out", type=str, default=None, help="Output directory for spectrogram images (default: <csv_out_dir>/spectros)")

    args = parser.parse_args([
        "--root", r"F:\BirdData2025",
        "--round_dir", "SpeedTest",  # Change to "2_round" for the other dataset
        "--model", r"C:\Users\admin\Desktop\thesisalgos\runs\detect\train9\weights\best.pt",
        "--csv_out", r"F:\BirdData2025\SpeedTest.csv",
        "--device", "cuda:0",
        "--imgsz", "640",
        "--conf", "0.7",
        "--iou", "0.45",
        "--segment_duration", str(SEGMENT_DURATION_DEFAULT),
        "--sr_target", str(SR_TARGET_DEFAULT)
    ])

    #args = parser.parse_args()

    img_out = Path(args.img_out) if args.img_out is not None else None

    run_pipeline(
        model_path=Path(args.model),
        root=Path(args.root),
        round_dir=args.round_dir,
        csv_out=Path(args.csv_out),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        segment_duration=args.segment_duration,
        sr_target=args.sr_target,
        batch=args.batch,
        half=args.half,
        img_out=img_out,
    )


if __name__ == "__main__":
    main()
