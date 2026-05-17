import argparse
import json
import os
from pathlib import PurePath

import matplotlib
import numpy as np
from PIL import Image, ImageDraw


def overlay_masks(image: np.ndarray | Image.Image, masks: np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGBA")

    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()

    masks = (255 * masks).astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image


def _normalize_path(value: str) -> str:
    return str(PurePath(value.replace("\\", "/"))).lower()


def _find_image_entry(images: list[dict], image_path: str) -> dict | None:
    target_path = _normalize_path(image_path)
    target_name = PurePath(target_path).name

    exact_matches = []
    basename_matches = []
    for image in images:
        file_name = image.get("file_name")
        if not file_name:
            continue

        normalized_file_name = _normalize_path(file_name)
        if normalized_file_name == target_path:
            exact_matches.append(image)
            continue

        if PurePath(normalized_file_name).name == target_name:
            basename_matches.append(image)

    if exact_matches:
        return exact_matches[0]

    if len(basename_matches) == 1:
        return basename_matches[0]

    if len(basename_matches) > 1:
        raise ValueError(
            f"Found multiple COCO entries matching '{target_name}'. "
            "Pass an image path that matches the stored relative path."
        )

    return None


def load_coco_masks(json_path: str, image_filename: str, image_size: tuple) -> np.ndarray:
    """Parses a COCO JSON file to generate mask arrays for a specific image."""
    with open(json_path) as f:
        coco_data = json.load(f)

    # 1. Find the image entry that matches our file. COCO exports may store either a
    image_entry = _find_image_entry(coco_data.get("images", []), image_filename)
    if image_entry is None:
        raise ValueError(f"Could not find an entry for '{image_filename}' in the COCO JSON.")
    image_id = image_entry["id"]

    # 2. Find all annotations for this image ID and convert them to masks
    masks = []
    for ann in coco_data.get("annotations", []):
        if ann["image_id"] == image_id:
            segmentation = ann.get("segmentation", [])

            if isinstance(segmentation, list) and len(segmentation) > 0:
                for poly in segmentation:
                    mask_img = Image.new("L", image_size, 0)
                    draw = ImageDraw.Draw(mask_img)
                    draw.polygon(poly, outline=1, fill=1)
                    masks.append(np.array(mask_img))

    if not masks:
        raise ValueError(f"No annotations found for '{image_filename}'.")

    # Stack into shape (N, H, W)
    return np.stack(masks)


def main():
    parser = argparse.ArgumentParser(description="Overlay mask annotations onto an image.")
    parser.add_argument("--image_path", type=str, help="Path to the input image file (e.g., .jpg, .png).")
    parser.add_argument("--annotation_path", type=str, help="Path to the annotation mask file (e.g., .npy or .json).")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at '{args.image_path}'")
        return

    if not os.path.exists(args.annotation_path):
        print(f"Error: Annotation file not found at '{args.annotation_path}'")
        return

    try:
        print(f"Loading image from {args.image_path}...")
        image = Image.open(args.image_path)

        print(f"Parsing COCO JSON from {args.annotation_path}...")
        image_filename = os.path.basename(args.image_path)
        masks = load_coco_masks(args.annotation_path, image_filename, image.size)

        print("Applying mask overlay...")
        result_image = overlay_masks(image, masks)

        result_image.convert("RGB").save("output_overlay.jpg")
        print("Saved output to output_overlay.jpg")

    except Exception as e:
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
