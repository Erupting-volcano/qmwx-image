"""Minimal COLMAP stand-in for tests: writes the files the backend parses.

Controlled through environment variables so tests never need the real engine.
"""

import os
from pathlib import Path
import sys
import time


def argument(name: str) -> str:
    values = sys.argv[2:]
    return values[values.index(name) + 1] if name in values else ""


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    command = sys.argv[1]
    if os.environ.get("FAKE_COLMAP_FAIL") == command:
        print(f"simulated failure in {command}", file=sys.stderr)
        return 1
    delay = float(os.environ.get("FAKE_COLMAP_SLEEP", "0"))
    if delay:
        time.sleep(delay)

    if command == "feature_extractor":
        Path(argument("--database_path")).write_bytes(b"db")
        return 0
    if command == "exhaustive_matcher":
        return 0
    if command == "mapper":
        if os.environ.get("FAKE_COLMAP_NO_MODEL"):
            return 0
        model = Path(argument("--output_path")) / "0"
        model.mkdir(parents=True, exist_ok=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (model / name).write_bytes(b"x")
        return 0
    if command == "model_converter":
        target = Path(argument("--output_path"))
        if argument("--output_type") == "TXT":
            target.mkdir(parents=True, exist_ok=True)
            model_name = os.environ.get("FAKE_COLMAP_CAMERA_MODEL", "PINHOLE")
            parameters = {
                "PINHOLE": "3200 3200 2016 1512",
                "SIMPLE_PINHOLE": "3200 2016 1512",
                "SIMPLE_RADIAL": "3200 2016 1512 0.01",
                "WEIRD": "1 2 3",
            }.get(model_name, "1 2 3 4")
            (target / "cameras.txt").write_text(
                f"# Camera list\n1 {model_name} 4032 3024 {parameters}\n", encoding="utf-8")
            (target / "images.txt").write_text("1 1 0 0 0 0 0 0 1 000.png\n\n", encoding="utf-8")
            (target / "points3D.txt").write_text("", encoding="utf-8")
        else:
            target.write_bytes(b"ply\nformat ascii 1.0\n")
        return 0
    if command == "model_analyzer":
        if os.environ.get("FAKE_COLMAP_BAD_ANALYZER"):
            print("unexpected analyzer output")
            return 0
        images = os.environ.get("FAKE_COLMAP_IMAGES", "3")
        print(f"Images: {images}")
        print("Points: 12480")
        print("Mean reprojection error: 0.83px")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
