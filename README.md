# Owl Call Detection from Audio (YOLO11 + Faster R-CNN)

In this project, long-form audio recordings are processed into **Mel spectrograms** and analyzed with **object detection** to localize and classify owl vocalisations. The end-to-end pipeline turns a folder of `.wav` files into a **CSV of detections**, and a companion notebook plots summaries from that CSV.

This repository is intentionally minimal: **the files in the root folder are the full workflow** (training + inference + plotting).

---

## Repository contents (flat structure)

- `run_yolo11_audio_pipeline_fast.py`  
  End-to-end pipeline for **YOLO11**:
  - audio → segmentation → Mel spectrograms → YOLO detections → `detections.csv`

- `faster r-cnn pipeline.py`  
  End-to-end pipeline for **Faster R-CNN (ResNet-50 backbone)**:
  - audio → segmentation → Mel spectrograms → Faster R-CNN detections → `detections.csv`

- `Training.ipynb`  
  Notebook for training and/or dataset preparation (YOLO11 and/or Faster R-CNN depending on your workflow).

- `Faster R-CNN.ipynb`  
  Notebook focused on training/evaluating Faster R-CNN (ResNet-50 backbone).

- `owl_detections_plots.ipynb`  
  Notebook to load the exported detection CSV and generate plots (site/hour/day summaries, etc.).

- `README.md`  
  This file.

---

## Requirements

- Python 3.10+ recommended
- A GPU is strongly recommended for training (inference can run on CPU but will be slower)
- Audio input: `.wav` files recommended

### Install dependencies

If you already have PyTorch installed for your CUDA setup, install the rest:

```bash
pip install ultralytics torchvision torchaudio librosa soundfile numpy pandas matplotlib opencv-python tqdm pyyaml
```

If you *don’t* have PyTorch yet, install it first following the official instructions for your OS/CUDA version, then run the command above.

---

## Quickstart (run detection on a folder of audio)

1) Put your audio files into a folder, for example:

```text
/path/to/audio_folder/
  file_001.wav
  file_002.wav
  ...
```

2) Choose a model pipeline:

### Option A — YOLO11 pipeline

```bash
python run_yolo11_audio_pipeline_fast.py
```

This script is typically configured by editing variables near the top (e.g., `INPUT_DIR`, `WEIGHTS_PATH`, `OUTPUT_CSV`, thresholds, spectrogram parameters). Open the file and set:

- input folder path
- model weights path (`.pt`)
- output CSV path
- confidence / IoU settings (if exposed in the script)
- spectrogram parameters (if exposed in the script)

### Option B — Faster R-CNN pipeline

> Note: the filename contains spaces, so quote it when running.

```bash
python "faster r-cnn pipeline.py"
```

Similarly, open the script and set:

- input folder path
- model weights path
- output CSV path
- thresholds and spectrogram parameters (if exposed)

---

## Output: detections CSV

Both pipelines generate a CSV containing one row per detected event (one detection = one bounding box/classification on the spectrogram).

Typical fields (your exact column names may vary depending on the script implementation):

- **audio file reference** (filename/path)
- **clip start time** (or timestamp derived from filename)
- **species / class label**
- **confidence score**
- **bounding box coordinates** (spectrogram coordinate system, if exported)
- optional metadata (site ID/date if parsed from filename or folder structure)

If you want to standardize filenames for downstream analysis, a useful pattern is:

```text
<site>_<YYYY-MM-DD>_<HH-MM-SS>.wav
```

Then the pipeline/plotting code can parse site + time consistently.

---

## Plot results from the CSV

Use the notebook:

- `owl_detections_plots.ipynb`

Typical workflow inside the notebook:

1. Set the path to the exported `detections.csv`
2. Load CSV into pandas
3. Aggregate detections by:
   - site
   - hour-of-day
   - date/day
4. Plot summaries (bar charts / stacked bars / time series)

Run it with Jupyter:

```bash
jupyter notebook
```

Open `owl_detections_plots.ipynb` and execute the cells.

---

## Training

Training is notebook-driven in this repo.

### YOLO11 training
Use:
- `Training.ipynb`

Typical steps (high-level):
1. Create/verify dataset structure and labels (YOLO format).
2. Configure model + training parameters.
3. Train and export `best.pt` for inference.

### Faster R-CNN (ResNet-50) training
Use:
- `Faster R-CNN.ipynb` (and/or relevant parts in `Training.ipynb`)

Typical steps (high-level):
1. Load dataset (often COCO-style annotations or a custom dataset class).
2. Train torchvision Faster R-CNN with ResNet-50 backbone.
3. Save model weights for use in `faster r-cnn pipeline.py`.

---

## Notes / troubleshooting

### No detections (or too many false positives)
- Adjust the confidence threshold (increase to reduce false positives).
- Confirm that training preprocessing matches inference preprocessing:
  - segment length
  - sample rate
  - Mel spectrogram parameters (FFT size, hop length, number of Mel bands)
- Check that your weights file matches the pipeline (YOLO weights for YOLO pipeline, Faster R-CNN weights for Faster R-CNN pipeline).

### Audio issues
- Prefer `.wav` input.
- If you have mixed sample rates, confirm the script resamples consistently.

### Speed
- YOLO is typically much faster than Faster R-CNN.
- For large batches, run on GPU and ensure torch/CUDA are installed correctly.

---

## License

Add your preferred open-source license (e.g., MIT) if you plan to share this publicly.
