# Aerial GCP Pose Estimation

Given a raw drone image, this pipeline returns the pixel centre of the Ground Control Point marker and its shape (`Cross`, `Square` or `L-Shape`).

Two stages, both a ResNet-34 with a heatmap output:

1. **Locate.** The full frame, resized to 1280×960, produces a heatmap whose top 3 peaks are candidate markers.
2. **Refine and classify.** A 384×384 crop at *full* resolution around each candidate gives a sub-pixel centre and the shape. The candidate with the best score wins.

`DECISIONS.md` records every design choice, the evidence behind it and the alternatives rejected.

## Layout
```
eda.py              data checks: label/image integrity, sizes, EXIF, class balance, leakage
gcp/data.py         indexing, label cleaning, project-level split, image cache
gcp/datasets.py     datasets and augmentation for both stages
gcp/model.py        HeatNet: timm backbone + FPN decoder (stride 4) + class head
gcp/heatmap.py      Gaussian targets, focal loss, sub-pixel peak decoding
gcp/pipeline.py     two-stage inference and metrics (PCK, macro F1)
train.py            training for either stage
infer.py            writes predictions.json, or evaluates on held-out projects
kaggle_run.ipynb    end-to-end runner for a Kaggle GPU notebook
predictions.json    predictions for the unlabelled test set
DECISIONS.md        decision log
```

## What the data turned out to be
`python eda.py --train <train_dir> --test <test_dir>` reports all of the following.

| Observation | Consequence |
|---|---|
| Images are 4096×2730, 4096×3068, and 4000×3000 for two test files, not the documented 2048×1365. | Everything works in original pixels; no input size is hard-coded. |
| Markers span roughly 15–80 px and appear anywhere in the frame (x from 67 to 3937). | The two-stage design: downscaling alone cannot reach 10 px accuracy, and full-resolution inference over a 12 MP frame is wasteful. |
| **Each project uses exactly one shape** (11 projects, 159 GCPs). | Validation holds out whole projects, one per class (RDCW, Egypt-New city, Deora). A random split would reward recognising the terrain instead of the marker. |
| **64 of the 73 test projects never appear in training.** | Unseen sites are what matter, hence strong scale/rotation/photometric augmentation and project-level model selection. |
| Class counts: L-Shape 491, Square 328, Cross 177. | Inverse-frequency class weights, since macro F1 weights every class equally. |
| 4 labels have no `verified_shape`. | Filled from other images of the same GCP; one physical marker has one shape. |
| 33 test images share a `gcp_id` folder with training labels, but no file is byte-identical. | Training never reads the test folder. At inference those images take the marker's known shape (`--no-gcp-prior` disables it). |
| The label file is `gcp_marks.json` and the data spells the class `L-Shape`, while the brief says `curated_gcp_marks.json` and `L-Shaped`. | The loader matches `*gcp_marks*.json`; output follows the data, and `--l-name` overrides it. |
| Delivered as multi-part Drive zips, each wrapping the tree in its own folder. | Paths are normalised from the `train_dataset` folder onward, so all 1000 labels resolve however the data is laid out. |
| No EXIF rotation, no duplicates, no out-of-bounds coordinates. | Nothing to correct. |

## Training
* **Loss.** Penalty-reduced focal loss on Gaussian heatmaps (σ = 1.5 cells in stage 1, 2 in stage 2). Stage 2 adds 0.5 × weighted cross-entropy, label smoothing 0.05.
* **Augmentation.** One affine warp moves image and keypoint together: arbitrary rotation, horizontal flip, scale 0.7–1.45 for altitude changes, and ±110 px crop jitter so stage 2 tolerates stage-1 error. Photometric jitter covers cameras, seasons and sun angle. All three shapes survive flips and rotations unchanged.
* **Negatives.** 30% of stage-2 samples are background crops with an empty target, which turns the stage-2 peak into a usable confidence for reranking stage-1 candidates.
* **Schedule.** AdamW, OneCycle at 1e-3, AMP, batch 8 (stage 1) and 32 (stage 2).
* **Speed.** Each 12 MP JPEG is decoded once into a canvas plus crops, so the GPU is not waiting on 4 CPU cores.
* **Test time.** Horizontal-flip averaging in both stages; shape probabilities averaged across each `gcp_id` folder.

## Results
Full pipeline on three held-out projects (141 images), none of them seen in training:

| PCK@10 | PCK@25 | PCK@50 | median px | macro F1 |
|---|---|---|---|---|
| 0.936 | 0.936 | 0.936 | 1.05 | 0.490 |

**Localisation.** The three thresholds are identical, so errors are all-or-nothing: 132 of 141 images are placed almost exactly (median 1.05 px, well inside the 10 px threshold) and 9 miss the marker outright. Nothing sits in between. Pushing the score further means recovering those misses, not sharpening the hits, which is why candidate reranking exists and where more work would go next.

**Classification** is the weaker half, at 0.49 against 0.33 for chance. Two causes, both visible in the data:
* Each training project uses a single shape, so the network can lean on what a *site* looks like. That cue is worthless on a new site, and this split deliberately measures that worst case.
* The measurement is coarse. Three held-out projects mean one class each, and per-GCP voting reduces the score to a handful of site-level decisions, so it moves in large steps.

On the test set both numbers should be better: 9 of its projects appear in training (133 of 300 images), and 33 images belong to GCPs whose shape is already known. The most promising next step is more site diversity rather than a bigger model, since 8 projects is thin for learning shape independently of terrain.

## Reproducing
**Weights.** `s1.pt` (stage 1) and `s2.pt` (stage 2), about 50 MB together:
[Google Drive](https://drive.google.com/drive/folders/1x9cou-St9HSgm8cfZQfgXRyJsEvMTGD8?usp=sharing). Place them beside the code.

```bash
pip install -r requirements.txt

# predictions on the test set
python infer.py --images <test_dataset> --s1 s1.pt --s2 s2.pt \
                --train-roots <train_dataset> --out predictions.json

# metrics on held-out projects
python infer.py --eval --train-roots <train_dataset> --s1 s1.pt --s2 s2.pt

# training (about 35 minutes on one T4)
python train.py --stage 1 --train-roots <train_dataset> --out s1.pt --epochs 20
python train.py --stage 2 --train-roots <train_dataset> --out s2.pt --epochs 25
```
`--train-roots` takes one or more folders and searches them recursively, so the multi-part downloads can be passed as they are. `--full` trains on every project once the epoch count is settled.

On Kaggle: upload this repo and the data as datasets, attach both to a notebook with a GPU and Internet enabled, and run `kaggle_run.ipynb`.

## Assumptions
* The hidden ground truth uses the training vocabulary (`L-Shape`).
* Every image contains exactly one marker, as is true throughout the training data, so the pipeline always emits one point.
* A `gcp_id` folder identifies one physical marker. The shape vote and the known-marker shortcut both rest on this, and it holds for all 159 training GCPs.
