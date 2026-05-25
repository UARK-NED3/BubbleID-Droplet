from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .pipeline import build_predictor, run_image
from .wsp import download_wsp_dataset, run_wsp_subset


def main() -> None:
    parser = argparse.ArgumentParser(prog="bubbleid-droplet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run tiled droplet segmentation on one image")
    add_common_args(predict)
    predict.add_argument("--image", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.add_argument(
        "--stitch-mode",
        choices=["instance-nms", "legacy-grid"],
        default="instance-nms",
        help="Use instance NMS tiling or the original BubbleID-Droplet shifted-grid mask stitching.",
    )

    wsp = subparsers.add_parser("evaluate-wsp", help="Run a subset of the Zenodo WSP dataset")
    add_common_args(wsp)
    wsp.add_argument("--dataset-dir")
    wsp.add_argument("--download-dir", default="data/wsp")
    wsp.add_argument("--output-dir", required=True)
    wsp.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    predictor = build_predictor(args)

    if args.command == "predict":
        summary = run_image(
            image_path=args.image,
            output_dir=args.output_dir,
            predictor=predictor,
            backend=args.backend,
            tile_size=args.tile_size,
            overlap=args.overlap,
            iou_threshold=args.iou_threshold,
            stitch_mode=args.stitch_mode,
            edge_margin=args.edge_margin,
        )
        print(json.dumps(asdict(summary), indent=2, default=str))
        return

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else download_wsp_dataset(args.download_dir)
    csv_path = run_wsp_subset(
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        predictor=predictor,
        backend=args.backend,
        limit=args.limit,
        tile_size=args.tile_size,
        overlap=args.overlap,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps({"results_csv": str(csv_path)}, indent=2))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["detectron2", "classical"], default="detectron2")
    parser.add_argument("--weights", help="Path to BubbleID Detectron2 .pth weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--min-size-test", type=int, default=800)
    parser.add_argument("--max-size-test", type=int, default=1333)
    parser.add_argument("--detections-per-image", type=int, default=3000)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--edge-margin", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-area", type=int, default=8)


if __name__ == "__main__":
    main()
