# Decision Log

Each entry: what was decided, why, and what was rejected. Evidence comes from `eda.py` on the provided data.

## D1. Predict in original full-resolution pixels, not the documented 2048×1365
The brief states 2048×1365, but the images are 4096×2730 (281 train / 178 test), 4096×3068 (324 / 120) and 4000×3000 (2 test). No code path assumes a fixed input size; every stage carries its own scale factor, and outputs are mapped back to the source image.
*Rejected:* resizing everything to 2048×1365, which would have halved effective resolution and put PCK@10 out of reach.

## D2. Two-stage coarse-to-fine, rather than one pass over the whole image
Markers occupy roughly 15–80 px in a 12 MP frame and can sit anywhere (x from 67 to 3937). A single downscaled pass cannot hit 10 px; a single full-resolution pass costs ~50× more compute per image.
Stage 1 sees the whole frame at 1280×960 and proposes locations; stage 2 sees a 384×384 full-resolution crop and produces the final point and the shape.
*Rejected:* tiled full-resolution inference (slow, many false positives per tile); direct (x, y) coordinate regression (weaker for small objects, no confidence signal to rank candidates).

## D3. Heatmaps with focal loss, decoded to sub-pixel accuracy
A stride-4 Gaussian heatmap with CenterNet's penalty-reduced focal loss handles the extreme foreground/background imbalance of a marker in a 12 MP image. The peak is refined by a weighted centroid over its 5×5 neighbourhood, which recovers the true point to within 0.85 px on a target heatmap.
*Rejected:* L2 regression on coordinates; argmax without refinement (quantises error to 4 px and caps PCK@10).

## D4. Validation holds out whole projects
Every training project uses a single shape (e.g. all 252 Vedanta images are L-Shape). A random image split would let a model score well on shape by recognising terrain, and would also leak, because multiple images show the same physical marker. Validation therefore holds out one project per class: RDCW (Cross), Egypt-New city (Square), Deora (L-Shape).
This also matches the test set: 64 of its 73 projects never appear in training, so unseen-site performance is what counts.
*Rejected:* random or stratified image split; k-fold (too costly for the time budget, and 11 projects make folds coarse).

## D5. Stage 2 trains on background crops and reranks candidates
Stage 1 offers its top 3 peaks; stage 2 scores each and the pipeline keeps the highest `s2_score · s1_score^0.25`. To make that score meaningful, 30% of stage-2 training samples are background crops with an empty target. This recovers images where stage 1's top peak lands on a bright rock or road marking.
*Rejected:* trusting stage 1's argmax (a single early mistake is unrecoverable and costs ~1/300 of PCK per image).

## D6. Augmentation built around what varies between sites
One affine warp moves image and keypoint together: arbitrary rotation (drone yaw is arbitrary), horizontal flip, and scale 0.7–1.45 (flight altitude and ground sample distance vary between surveys). All three shapes are unchanged by flips and rotations, so labels stay valid. Photometric augmentation (brightness, contrast, saturation, gamma, blur, noise) covers different cameras, seasons and sun angles. Stage-2 crops are jittered by ±110 px (occasionally ±220 px) so the model tolerates stage-1 error.
*Rejected:* vertical-only flips and small rotations, which would underuse the rotational symmetry of the task.

## D7. Inverse-frequency class weights
Class counts are L-Shape 491, Square 328, Cross 177. Macro F1 weights each class equally, so the loss does too, with label smoothing 0.05.
*Rejected:* plain cross-entropy (biases towards L-Shape); resampling (would distort the keypoint task sharing the same batch).

## D8. Shape decided per GCP folder, not per image
All images inside one `project/survey/gcp_id` folder show the same physical marker, so shape probabilities are averaged across the folder before the argmax. This fixes single-image mistakes on blurred or shadowed frames.
The same fact fills in the 4 training labels that have no `verified_shape`, using the other images of the same GCP.
*Rejected:* independent per-image classification (noisier, and ignores structure that production already provides).

## D9. Known markers reuse their labelled shape
33 test images live in a `gcp_id` folder that also appears in the training labels. They are different photographs (no byte-identical files), so this is not leakage from the test set; it is the same surveyed marker, whose shape is already known. Those images take the labelled shape. `--no-gcp-prior` turns this off.
*Rejected:* ignoring the correspondence, which would throw away certain information a production pipeline would have.

## D10. Multi-part downloads and path handling treated as a first-class problem
The data arrives as multi-part zips, each wrapping the tree in its own folder, and training environments mount it under their own prefix. Keys are normalised from the `train_dataset`/`test_dataset` folder onward and wrapper folders are dropped, so all 1000 labels resolve regardless of how the data is laid out. Loading training data excludes the `test_dataset` folder, because 33 relative paths exist in both. The loader fails loudly if no label matches an image, rather than silently training on nothing.

## D11. ResNet-34 rather than a heavier backbone
The brief values practical, deployable solutions. ResNet-34 with a light FPN decoder is plain PyTorch with no custom ops; both checkpoints together are about 50 MB, and the whole run fits in the free Kaggle GPU budget. The backbone is a command-line flag, so a larger one can be swapped in if accuracy matters more than cost.
*Rejected:* HRNet or a ViT pose model (slower, heavier, and unnecessary for a single keypoint); YOLO-pose (an extra framework dependency for one keypoint and three classes).

## D12. Output labels use the dataset's spelling
The training JSON says `L-Shape`, the brief says `L-Shaped`. Predictions follow the data on the assumption that the hidden ground truth comes from the same source; `--l-name "L-Shaped"` switches it in one flag.

## D13. Ship the validated models, not an unvalidated retrain
The plan was to settle the epoch count on held-out projects and then retrain on all 1000 images. Within the time available only one run was possible, so the delivered models are the validated ones, trained on 859 images from 8 projects. Every number in the README therefore comes from these exact weights rather than from a sibling run.
*Rejected:* retraining on all projects and shipping that instead, which would probably classify a little better (more site diversity) but would leave no measured evidence for the weights actually submitted. `--full` runs it when the extra data is worth more than the measurement.
