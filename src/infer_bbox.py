import os
import argparse
import json
from pathlib import Path, PurePath
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.load_checkpoint import get_sam_vit_t
from src.transform import MEAN, STD

LABEL_MAPS = {
    "coffee": {
        0: "Benh sau ve bua",
        1: "Benh phan trang",
        2: "Benh nam ri sat",
        3: "Benh dom rong",
    },
    "rice": {
        0: "BrownSpot",
        1: "Healthy",
        2: "Hispa",
        3: "LeafBlast",
    },
}

CLASS_COLORS = {
    0: (0, 0, 255),      # Red for BrownSpot / Benh sau ve bua
    1: (0, 255, 0),      # Green for Healthy / Benh phan trang
    2: (255, 0, 0),      # Blue for Hispa / Benh nam ri sat
    3: (0, 255, 255),    # Yellow for LeafBlast / Benh dom rong
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run bounding-box-guided SAM class-segmentation inference.")
    parser.add_argument("--input", type=str, required=True, help="Path to single image or input directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to directory to save results.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MobileSAM checkpoint.")
    parser.add_argument("--dataset", type=str, default="rice", choices=["rice", "coffee"], help="Dataset name.")
    parser.add_argument("--bbox", type=int, nargs=4, default=None, help="Optional custom bbox [xmin ymin xmax ymax].")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--img-size", type=int, default=1024, help="Model input image size.")
    return parser.parse_args()

def find_annotations_for_image(dataset_dir, image_path):
    """Try to find annotation file and extract ground-truth bboxes for the image."""
    annotation_path = Path(dataset_dir) / "annotations.coco.json"
    if not annotation_path.exists():
        annotation_path = Path(dataset_dir).parent / "annotations.coco.json"
        if not annotation_path.exists():
            return []

    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load annotation file: {e}")
        return []

    target_name = PurePath(image_path).name.lower()
    img_id = None
    for img in coco_data.get("images", []):
        file_name = img.get("file_name", "")
        if PurePath(file_name).name.lower() == target_name:
            img_id = img["id"]
            break

    if img_id is None:
        return []

    bboxes = []
    for ann in coco_data.get("annotations", []):
        if ann["image_id"] == img_id:
            x, y, w, h = ann["bbox"]
            bboxes.append([int(x), int(y), int(x + w), int(y + h)])
    return bboxes

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = get_sam_vit_t(
        checkpoint_path=args.checkpoint,
        resume=False,
        num_mask_outputs=4,
        allow_download=False,
        strict_checkpoint_shapes=True,
    )
    model.to(device)
    model.eval()

    # Determine input images
    if os.path.isdir(args.input):
        img_paths = list(Path(args.input).rglob("*.jpg")) + list(Path(args.input).rglob("*.jpeg"))
        dataset_dir = args.input
    else:
        img_paths = [Path(args.input)]
        dataset_dir = str(Path(args.input).parent)

    os.makedirs(args.output, exist_ok=True)
    labels_map = LABEL_MAPS[args.dataset]

    for img_path in img_paths:
        print(f"\nProcessing: {img_path}")
        image_cv = cv2.imread(str(img_path))
        if image_cv is None:
            print(f"Error: Could not load image {img_path}")
            continue

        h_orig, w_orig = image_cv.shape[:2]

        # Determine bounding boxes to run on. Default to full image
        bboxes = []
        if args.bbox is not None:
            bboxes = [args.bbox]
            print(f"Using custom bbox: {bboxes[0]}")
        else:
            bboxes = find_annotations_for_image(dataset_dir, str(img_path))
            if bboxes:
                print(f"Found {len(bboxes)} ground-truth bbox(es) in annotations.")
            else:
                bboxes = [[0, 0, w_orig, h_orig]]
                print("No annotations found. Defaulting to full image bounding box.")

        # Prepare normalized image tensor for SAM (1, 3, 1024, 1024)
        image_resized = cv2.resize(image_cv, (args.img_size, args.img_size))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).float() / 255.0
        
        # Apply normalization
        mean = torch.tensor(MEAN).view(-1, 1, 1)
        std = torch.tensor(STD).view(-1, 1, 1)
        image_tensor = ((image_tensor - mean) / std).unsqueeze(0).to(device)

        vis_image = image_cv.copy()

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            # Clamp coordinates to image boundaries
            x_min = max(0, min(x_min, w_orig))
            x_max = max(0, min(x_max, w_orig))
            y_min = max(0, min(y_min, h_orig))
            y_max = max(0, min(y_max, h_orig))

            if x_max <= x_min or y_max <= y_min:
                continue

            # Scale bounding box to 1024x1024
            scale_x = args.img_size / w_orig
            scale_y = args.img_size / h_orig
            scaled_bbox = [
                x_min * scale_x,
                y_min * scale_y,
                x_max * scale_x,
                y_max * scale_y
            ]

            bbox_tensor = torch.tensor(scaled_bbox).float().unsqueeze(0).to(device) # (1, 4)

            with torch.no_grad():
                pred_masks, pred_IOUs = model(image_tensor, bbox_tensor)

            # Classify by picking channel with highest predicted IoU score
            pred_class_id = torch.argmax(pred_IOUs[0]).item()
            pred_score = pred_IOUs[0, pred_class_id].item()
            class_name = labels_map.get(pred_class_id, f"Class {pred_class_id}")
            print(f"Box {bbox} -> Predicted Class: {class_name} (Score: {pred_score:.4f})")

            # Extract the predicted mask for the class
            mask_1024 = (torch.sigmoid(pred_masks[0, pred_class_id]) > 0.5).cpu().numpy().astype(np.uint8)
            
            # Upscale mask back to original resolution
            mask_orig = cv2.resize(mask_1024, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # Color overlay
            color = CLASS_COLORS.get(pred_class_id, (0, 255, 0))
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask_orig > 0] = color
            
            # Weighted overlay on original image
            vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.4, 0)

            # Draw bounding box
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw class name and confidence score
            label_text = f"{class_name} ({pred_score:.2f})"
            cv2.putText(
                vis_image,
                label_text,
                (x_min, max(y_min - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Save visual result
        img_name = img_path.stem
        save_dir = Path(args.output) / img_name
        save_dir.mkdir(parents=True, exist_ok=True)
        vis_path = save_dir / "visualization.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        print(f"Saved visualization to {vis_path}")

    print("PyTorch BBox Inference completed!")

if __name__ == "__main__":
    main()
