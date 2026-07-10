#!/usr/bin/env python3
"""
Split large images in inter_patch_tif/ into 256x256 patches
and generate per-image test.jsonl files.

Output structure for each sample:
  sample/
    lane_ins_png/{big_image_stem}/
      r{row:03d}_c{col:03d}.png
    test/{big_image_stem}.jsonl
"""

import argparse
import json
import tarfile
import sys
from pathlib import Path

from PIL import Image


DEFAULT_PROMPT = "<image>\nPlease construct the complete road map in the current BEV (Bird's Eye View) image patch.\nCoordinates use a normalized 0-1000 grid over the original 256x256 image patch.\n\nIncoming traces JSON:\n[]\n\nIncoming intersections JSON:\n[]"

def safe_extract_tar_gz(archive_path):
    """Extract a .tar.gz archive, stripping a single top-level directory if present."""
    import shutil
    import tempfile

    target_dir = archive_path.with_suffix("").with_suffix("").resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                member_path = (tmp_path / member.name).resolve()
                try:
                    member_path.relative_to(tmp_path)
                except ValueError:
                    raise ValueError(f"unsafe archive member path: {member.name}")
            tar.extractall(path=str(tmp_path))

        contents = sorted(tmp_path.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            src = contents[0]
        else:
            src = tmp_path

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            shutil.move(str(item), str(target_dir / item.name))

    return target_dir


def extract_archives(root, delete_after=False):
    """Scan root for .tar.gz files and extract each."""
    archives = sorted(root.glob("*.tar.gz"))
    if not archives:
        return
    print(f"Found {len(archives)} .tar.gz archives, extracting...")
    for archive in archives:
        target = safe_extract_tar_gz(archive)
        print(f"  {archive.name} -> {target.name}/")
        if delete_after:
            archive.unlink()



def pad_to_multiple(image, patch_size):
    """Pad image so dimensions are multiples of patch_size (zero-padding)."""
    w, h = image.size
    pad_w = (-w) % patch_size
    pad_h = (-h) % patch_size
    if pad_w == 0 and pad_h == 0:
        return image
    result = Image.new(image.mode or "RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    result.paste(image, (0, 0))
    return result


def split_one_image(image_path, output_dir, patch_size=256, stride=None, min_blue_pixels=10):
    """Split a single large image into patches, save to output_dir.
    Returns (patches, original_size, padded_size)."""
    if stride is None:
        stride = patch_size

    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    original_size = img.size
    img = pad_to_multiple(img, patch_size)
    padded_size = img.size
    w, h = padded_size

    patches = []
    row = 0
    y = 0
    while y + patch_size <= h:
        col = 0
        x = 0
        while x + patch_size <= w:
            patch = img.crop((x, y, x + patch_size, y + patch_size))

            if min_blue_pixels > 0:
                _, _, b = patch.split()
                hist = b.histogram()
                blue_nonzero = sum(hist[1:])
                if blue_nonzero < min_blue_pixels:
                    continue

            name = f"r{row:03d}_c{col:03d}.png"
            patch.save(output_dir / name, format="PNG")
            patches.append((row, col))
            col += 1
            x = col * stride
        row += 1
        y = row * stride

    return patches, original_size, padded_size


def generate_jsonl(output_dir, dataset_root, sample_id, big_image_stem, patches, prompt,
                   original_image_size=None, padded_image_size=None):
    """Generate a jsonl file for a single big image's patches."""
    jsonl_dir = output_dir / "rc_one_patch_release/center_line_v2/test"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{big_image_stem}.jsonl"

    src_w, src_h = padded_image_size or (0, 0)
    orig_w, orig_h = original_image_size or src_w, src_h

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row, col in patches:
            x0 = col * 256
            y0 = row * 256
            patch_name = f"r{row:03d}_c{col:03d}.png"
            image_rel = f"{sample_id}/rc_one_patch_release/center_line_v2/lane_ins_png/{big_image_stem}/{patch_name}"

            meta = {
                "tile_id": big_image_stem,
                "log_id": big_image_stem,
                "patch_row": row,
                "patch_col": col,
                "row": row,
                "col": col,
                "x0": x0,
                "y0": y0,
                "patch_size": 256,
                "stride": 256,
                "source_image_size": [src_w, src_h],
                "original_source_image_size": [orig_w, orig_h],
                "coord_system": "patch_norm1000",
                "task_mode": "state_update_centerline_intersection",
                "raw_sample_root": str(output_dir),
                "scan_order": "row_major_top_to_bottom_left_to_right",
                "available_neighbors": ["left", "top"],
                "train_shuffle_allowed": True,
                "trace_source_train": "none",
                "trace_source_infer": "predicted_left_top_neighbors",
                "phase": "phase_a",
                "coord_mode": "norm1000",
                "coord_range": 1000,
                "pixel_patch_size": 256,
                "patch_width": 256,
                "patch_height": 256,
                "intersection_hint_source_train": "none",
            }

            record = {
                "id": f"{sample_id}_{big_image_stem}_{patch_name.replace('''.png''', '''''')}",
                "image": image_rel,
                "meta": meta,
                "conversations": [
                    {"from": "human", "value": prompt}
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return jsonl_path



def process_dataset(dataset_root, patch_size=256, stride=None, prompt=DEFAULT_PROMPT, extract=False, min_blue_pixels=10):
    """Scan dataset/ and process every sample with inter_patch_tif/."""
    root = Path(dataset_root)
    img_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    if extract:
        extract_archives(root, delete_after=False)

    sample_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "rc_one_patch_release/center_line_v2/inter_patch_tif").is_dir()
    )

    if not sample_dirs:
        print(f"Error: no sample directories (with inter_patch_tif/) found under {root}")
        sys.exit(1)

    total_patches = 0
    total_images = 0

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        infer_dir = sample_dir / "rc_one_patch_release/center_line_v2/inter_patch_tif"
        image_files = sorted(
            p for p in infer_dir.iterdir()
            if p.is_file() and p.suffix.lower() in img_extensions
        )

        if not image_files:
            print(f"  [SKIP] {sample_id}/inter_patch_tif/ - no image files found")
            continue

        for img_path in image_files:
            big_stem = img_path.stem
            out_dir = sample_dir / "rc_one_patch_release/center_line_v2/lane_ins_png" / big_stem

            patches, orig_size, pad_size = split_one_image(img_path, out_dir, patch_size, stride, min_blue_pixels)
            jsonl_path = generate_jsonl(
                sample_dir, root, sample_id, big_stem, patches, prompt,
                original_image_size=orig_size, padded_image_size=pad_size
            )

            print(
                f"  {sample_id}/{img_path.name} -> "
                f"lane_ins_png/{big_stem}/  ({len(patches)} patches)"
            )
            print(f"    jsonl: {jsonl_path.relative_to(root.parent)}")
            total_patches += len(patches)
            total_images += 1

    print(
        f"\nDone: {total_images} large images -> {total_patches} patches "
        f"in {len(sample_dirs)} samples"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split raw large images into patches and generate test.jsonl"
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Root directory containing sample folders (default: dataset)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=256, help="Patch size (default: 256)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between patches (default: equal to patch_size, no overlap)",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Extract .tar.gz archives in dataset-root before processing",
    )
    parser.add_argument(
        "--min-blue-pixels", type=int, default=10,
        help="Skip patches with fewer non-zero blue pixels (default: 10, 0=disable)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Inference prompt for test.jsonl conversations",
    )
    args = parser.parse_args()
    process_dataset(args.dataset_root, args.patch_size, args.stride, args.prompt, args.extract, args.min_blue_pixels)


if __name__ == "__main__":
    main()
