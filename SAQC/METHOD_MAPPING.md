# SAQC paper-to-project mapping

## Paper-faithful pieces

- Local evidence is a zero-padded `7 x 7` BEV feature patch.
- Two relative row/column offset channels are concatenated to the patch.
- LSQH uses two `3 x 3` convolutions with ReLU, average pooling and a scalar quality output in `[0,1]`.
- Localization-quality supervision is matched BEV IoU with Smooth-L1 loss.
- Quality-aware ranking uses `s' = s * q^beta`; the paper's default operating point `beta=1.5` is the default here.
- Optional post-hoc calibration uses `sigmoid(a * logit(s') + b)` with soft IoU targets.
- SAQC never changes decoded 3D box geometry.

## Required adaptation for this repository

The paper uses an SFA3D-style center-based detector and explicitly states that
its current form is not architecture-agnostic. This repository uses an
anchor-based PointPillar/Where2comm detector, so there is no paper-defined
center heatmap cell `u_i` for a decoded box.

The primary adapter here is:

1. use the exact fused feature map consumed by the frozen F `cls_head` and `reg_head`;
2. decode the candidate with the existing OpenCOOD postprocessor;
3. map the decoded `(x,y)` center to the nearest BEV grid center from the frozen anchor grid;
4. crop the SAQC patch at that cell;
5. record whether that decoded-center cell equals the original anchor cell and record the center-to-grid metric distance.

This is an **anchor-based SAQC adaptation**, not an exact reproduction of the
paper's center-based detector implementation.

## Hidden width

The accessible paper text specifies the two-convolution LSQH structure but
does not provide an official source implementation in the repository. The
hidden width is therefore an explicit experiment parameter
(`--hidden-channels`, default 16) and is written into every checkpoint.
It must not be presented as a verified paper constant.

## Training protocol

- Frontend, GSPR, F fusion arm, detector heads and decoded boxes are frozen.
- Only LSQH is trained.
- This project's shared LSQH trains on complete official OPV2V train under Clean + once-materialized physics Fog/Rain/Snow. This freezes the epoch-0 weather realization; it is an experiment-protocol adaptation, not a SAQC paper detail.
- Positive anchor labels (`pos_equal_one > 0`) supply the anchor-based counterpart of the paper's positive matched quality samples.
- The quality target is the decoded positive anchor's maximum BEV IoU to frame GT.
- Validation/test GT never enters inference features or score construction.

The paper also describes detector/quality training stages. For this benchmark,
joint detector fine-tuning is deliberately not the primary protocol because
the scientific comparison is against a fixed-geometry GT-IoU Oracle. Changing
detector geometry would invalidate that controlled comparison.

## Evaluation protocol

Primary reliability benchmark:

- frozen F detector;
- geometry-valid top256 candidate pool selected by original F score before SAQC;
- unchanged decoded boxes;
- original rotated NMS and range filter;
- fixed per-frame final output budget equal to original F;
- compare original F, scorepass SAQC, top256 raw F score, top256 SAQC and top256 GT-IoU Oracle;
- report historical frame-order AP and cross-frame global-sort AP;
- report score/IoU Spearman, Q-ECE-style soft calibration error, and Oracle recovery ratio.

Development uses OPV2V validation with the fixed physics weather PCDs. Frozen final
benchmark uses OPV2V clean test and existing OPV2V-W fog/rain/snow test with
online weather augmentation disabled.
