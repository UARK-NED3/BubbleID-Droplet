from __future__ import annotations

import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .pipeline import run_image

WSP_URL = "https://zenodo.org/records/13995950/files/Dataset%20WSP.zip?download=1"


def download_wsp_dataset(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "Dataset_WSP.zip"
    if not archive.exists():
        with urllib.request.urlopen(WSP_URL) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
    extract_dir = destination / "Dataset_WSP"
    if not extract_dir.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    return extract_dir


def find_wsp_images(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    image_dirs = [p for p in dataset_dir.rglob("*") if p.is_dir() and p.name.lower() == "image"]
    roots = image_dirs if image_dirs else [dataset_dir]
    images: list[Path] = []
    for root in roots:
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            images.extend(root.rglob(pattern))
    return sorted(set(images))


def run_wsp_subset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    predictor,
    backend: str,
    limit: int,
    tile_size: int,
    overlap: int,
    iou_threshold: float,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = find_wsp_images(dataset_dir)[:limit]
    csv_path = output_dir / "wsp_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "droplet_count",
                "mean_diameter_px",
                "std_diameter_px",
                "backend",
            ],
        )
        writer.writeheader()
        for image in images:
            summary = run_image(
                image_path=image,
                output_dir=output_dir / image.stem,
                predictor=predictor,
                backend=backend,
                tile_size=tile_size,
                overlap=overlap,
                iou_threshold=iou_threshold,
            )
            writer.writerow(
                {
                    "image": str(image),
                    "droplet_count": summary.droplet_count,
                    "mean_diameter_px": f"{summary.mean_diameter_px:.6f}",
                    "std_diameter_px": f"{summary.std_diameter_px:.6f}",
                    "backend": summary.backend,
                }
            )
    return csv_path
