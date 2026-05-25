<div align="center">
  <img src="./Images/bubbleid-dropheader.png" alt="BubbleID-Droplet" style="width: 85%; max-width: 100%;">
</div>

# BubbleID-Droplet

BubbleID-Droplet adapts the original [BubbleID](https://github.com/cldunlap73/BubbleID)
Mask R-CNN model for dense droplet segmentation. The key change is tiled inference:
large droplet images are split into tiles, the BubbleID Detectron2 model is run on each
tile, predictions are mapped back into full-image coordinates, and droplet diameters are
exported from the final mask.

Two stitching modes are available:

- `legacy-grid`: reproduces the original BubbleID-Droplet notebook strategy with eight
  shifted grid passes, edge-touching detections removed, binary mask union, and watershed
  splitting. This is the recommended mode for matching the published droplet examples.
- `instance-nms`: a cleaner single-pass overlapping tile mode that preserves model
  instances and merges duplicates with mask IoU NMS. It is easier to reason about, but
  may under-detect dense condensation images compared with the legacy grid.

## Why Tiling Is Needed

Detectron2's `DefaultPredictor` resizes images at test time using
`INPUT.MIN_SIZE_TEST` and `INPUT.MAX_SIZE_TEST`, and standard configs also cap returned
detections through `TEST.DETECTIONS_PER_IMAGE`. On a dense full-frame condensation image,
small droplets can become too small in network coordinates and thousands of droplets
compete for a limited detection budget. Tiling keeps droplets at a scale closer to the
BubbleID training distribution and gives each tile its own detection budget.

## Install

The classical backend only needs the packages in `requirements.txt` and is intended for
smoke tests. The BubbleID model backend requires a working PyTorch + Detectron2
environment and the original BubbleID model weights.

```powershell
python -m pip install -r requirements.txt
```

Install PyTorch and Detectron2 separately for your CUDA, OS, and Python version. On
Windows this is often easiest in a Python 3.10 or 3.11 environment.

## Run BubbleID Tiled Inference

```powershell
python -m bubbleid_droplet.cli predict `
  --backend detectron2 `
  --weights C:\path\to\bubbleid_model.pth `
  --image C:\Users\hanhu\Downloads\Adera_Lab_Droplet.tif `
  --output-dir outputs\adera `
  --stitch-mode legacy-grid `
  --num-classes 2 `
  --min-size-test 800 `
  --max-size-test 1333 `
  --detections-per-image 3000 `
  --score-threshold 0.10
```

For a faster, instance-preserving experimental mode:

```powershell
python -m bubbleid_droplet.cli predict `
  --backend detectron2 `
  --weights C:\path\to\bubbleid_model.pth `
  --image C:\Users\hanhu\Downloads\Adera_Lab_Droplet.tif `
  --output-dir outputs\adera_instance_nms `
  --stitch-mode instance-nms `
  --tile-size 512 `
  --overlap 128 `
  --detections-per-image 3000 `
  --score-threshold 0.10
```

Outputs include:

- `*_fullmask.png`: binary visualization mask
- `*_labels.tif`: instance label mask
- `*_overlay.png`: input image with equivalent-diameter circles
- `*_diameters.txt`: equivalent droplet diameters in pixels

## Smoke Test Without Detectron2

The local fallback backend is not the BubbleID model. It exists to test image loading,
tiling, output writing, and WSP dataset plumbing on machines without Detectron2.

```powershell
python -m bubbleid_droplet.cli predict `
  --backend classical `
  --image C:\Users\hanhu\Downloads\Adera_Lab_Droplet.tif `
  --output-dir outputs\adera_classical
```

## Zenodo WSP Dataset Test

The WSP dataset is useful for testing dense droplet tiling and merging. It is not a
condensation microscopy dataset, but it includes instance polygon annotations and many
dense droplet scenes.

```powershell
python -m bubbleid_droplet.cli evaluate-wsp `
  --backend detectron2 `
  --weights C:\path\to\bubbleid_model.pth `
  --download-dir data\wsp `
  --output-dir outputs\wsp `
  --limit 25
```

For a fast plumbing check:

```powershell
python -m bubbleid_droplet.cli evaluate-wsp `
  --backend classical `
  --download-dir data\wsp `
  --output-dir outputs\wsp_classical `
  --limit 3
```

## Provenance

This repository is an adaptation of BubbleID for droplet images. It keeps the original
notebook prototype for reference and adds a reusable tiled-inference package for testing
and further development.
