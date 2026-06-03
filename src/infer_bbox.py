import argparse
import json
import os
from pathlib import Path, PurePath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch

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
    0: (0, 0, 255),  # Red for BrownSpot / Benh sau ve bua
    1: (0, 255, 0),  # Green for Healthy / Benh phan trang
    2: (255, 0, 0),  # Blue for Hispa / Benh nam ri sat
    3: (0, 255, 255),  # Yellow for LeafBlast / Benh dom rong
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run bounding-box-guided SAM class-segmentation inference.")
    parser.add_argument("--input", type=str, required=True, help="Path to single image or input directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to directory to save results.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MobileSAM checkpoint.")
    parser.add_argument(
        "--onnx", type=str, default=None, help="Path to exported ONNX model of SAM prompt encoder + mask decoder."
    )
    parser.add_argument("--dataset", type=str, default="rice", choices=["rice", "coffee"], help="Dataset name.")
    parser.add_argument("--bbox", type=int, nargs=4, default=None, help="Optional custom bbox [xmin ymin xmax ymax].")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on."
    )
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
        with open(annotation_path, encoding="utf-8") as f:
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

    ort_session = None
    if args.onnx is not None:
        print(f"Loading ONNX model from {args.onnx}...")
        providers = ["CPUExecutionProvider"]
        if args.device == "cuda" and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider"] + providers
        ort_session = onnxruntime.InferenceSession(args.onnx, providers=providers)

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
            scaled_bbox = [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]

            bbox_tensor = torch.tensor(scaled_bbox).float().unsqueeze(0).to(device)

            if ort_session is not None:
                # 1. Run the PyTorch Image Encoder to get image embeddings
                with torch.no_grad():
                    image_embeddings = model.image_encoder(image_tensor)

                # 2. Format inputs for ONNX model
                image_embeddings_np = image_embeddings.cpu().numpy()
                point_coords_np = np.array(
                    [[[scaled_bbox[0], scaled_bbox[1]], [scaled_bbox[2], scaled_bbox[3]]]], dtype=np.float32
                )
                point_labels_np = np.array([[2, 3]], dtype=np.float32)
                mask_input_np = np.zeros((1, 1, 256, 256), dtype=np.float32)
                has_mask_input_np = np.array([0.0], dtype=np.float32)
                orig_im_size_np = np.array([args.img_size, args.img_size], dtype=np.float32)

                ort_inputs = {
                    "image_embeddings": image_embeddings_np,
                    "point_coords": point_coords_np,
                    "point_labels": point_labels_np,
                    "mask_input": mask_input_np,
                    "has_mask_input": has_mask_input_np,
                    "orig_im_size": orig_im_size_np,
                }

                # 3. Run ONNX model
                ort_outputs = ort_session.run(None, ort_inputs)
                # Slice to remove the single mask prediction (index 0) to align with PyTorch multimask output
                pred_masks_np = ort_outputs[0][:, 1:, :, :]  # Shape: (1, 4, img_size, img_size)
                pred_IOUs_np = ort_outputs[1][:, 1:]  # Shape: (1, 4)

                # Classify by picking channel with highest predicted IoU score
                pred_class_id = np.argmax(pred_IOUs_np[0])
                pred_score = pred_IOUs_np[0, pred_class_id]
                class_name = labels_map.get(pred_class_id, f"Class {pred_class_id}")
                print(f"Box {bbox} -> Predicted Class (ONNX): {class_name} (Score: {pred_score:.4f})")

                # Sigmoid and thresholding to get binary mask of shape (img_size, img_size)
                mask_1024 = (1 / (1 + np.exp(-pred_masks_np[0, pred_class_id])) > 0.5).astype(np.uint8)

                # Upscale mask back to original resolution
                mask_orig = cv2.resize(mask_1024, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            else:
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
            cv2.putText(vis_image, label_text, (x_min, max(y_min - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save visual result
        img_name = img_path.stem
        save_dir = Path(args.output) / img_name
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.onnx is not None:
            vis_filename = f"visualization_{Path(args.onnx).stem}.jpg"
        else:
            vis_filename = "visualization.jpg"
        vis_path = save_dir / vis_filename

        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original image", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(vis_rgb)
        axes[1].set_title("Overlaid image", fontsize=14)
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(str(vis_path), bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved visualization to {vis_path}")

    if args.onnx is not None:
        print("ONNX BBox Inference completed!")
    else:
        print("PyTorch BBox Inference completed!")


if __name__ == "__main__":
    main()
