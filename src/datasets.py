import json
from pathlib import Path, PurePath

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


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


class SAMDataset(Dataset):
    """
    SAMDataset is a custom dataset class for images and their corresponding instance masks.
    Use COCO format JSON annotations. Yields individual instances with 4-channel masks.
    """

    def __init__(
        self,
        root_dir,
        annotation_path=None,
        transform=None,
        max_bbox_shift=10,
        split="train",
    ):
        """
        Args:
            root_dir (string): Directory containing images.
            annotation_path (string, optional): Path to COCO JSON annotations.
            transform (tuple, optional): A tuple of two optional transforms.
            max_bbox_shift (int, optional): Random perturbation to bounding box.
            split (str): "train", "val", or "test" (70/10/20 split).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_bbox_shift = max_bbox_shift
        self.split = split

        if annotation_path is None:
            if (self.root_dir / "annotations.coco.json").exists():
                annotation_path = self.root_dir / "annotations.coco.json"
            elif (self.root_dir.parent / "annotations.coco.json").exists():
                annotation_path = self.root_dir.parent / "annotations.coco.json"
            else:
                raise FileNotFoundError(f"Could not find annotations.coco.json in {self.root_dir}")

        self.annotation_path = Path(annotation_path)

        with open(self.annotation_path) as f:
            self.coco_data = json.load(f)

        self.images_data = self.coco_data.get("images", [])

        # Build mapping from image_id to image info
        self.image_info = {img["id"]: img for img in self.images_data}

        # Get all .jpg and .jpeg files
        all_images = list(self.root_dir.rglob("*.jpg")) + list(self.root_dir.rglob("*.jpeg"))
        
        # Match files to COCO entries
        valid_image_ids = []
        self.img_id_to_path = {}
        for img_path in all_images:
            entry = _find_image_entry(self.images_data, str(img_path))
            if entry is not None:
                img_id = entry["id"]
                valid_image_ids.append(img_id)
                self.img_id_to_path[img_id] = img_path
            else:
                print(f"Warning: {img_path} not found in COCO JSON!")

        # Deduplicate and sort for deterministic splitting
        valid_image_ids = sorted(list(set(valid_image_ids)))
        
        # Split logic (70/10/20)
        import random
        rng = random.Random(42)
        rng.shuffle(valid_image_ids)
        
        n_total = len(valid_image_ids)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)
        
        if self.split == "train":
            split_ids = valid_image_ids[:n_train]
        elif self.split == "val":
            split_ids = valid_image_ids[n_train:n_train+n_val]
        else: # test
            split_ids = valid_image_ids[n_train+n_val:]
            
        split_ids_set = set(split_ids)

        # Collect instances for the split
        self.instances = []
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in split_ids_set and img_id in self.img_id_to_path:
                self.instances.append({
                    "image_id": img_id,
                    "annotation": ann
                })

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        img_id = instance["image_id"]
        ann = instance["annotation"]
        img_name = self.img_id_to_path[img_id]

        image = Image.open(img_name).convert("RGB")
        image_size = image.size  # (width, height)
        image_np = np.array(image)

        # Generate 4-channel mask
        # Assuming category_id in 1, 2, 3, 4
        category_id = ann.get("category_id", 1)
        channel_idx = category_id - 1
        if channel_idx < 0 or channel_idx > 3:
            channel_idx = 0 # fallback
            
        mask_np = np.zeros((4, image_size[1], image_size[0]), dtype=np.uint8)
        
        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, list) and len(segmentation) > 0:
            for poly in segmentation:
                mask_img = Image.new("L", image_size, 0)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon(poly, outline=1, fill=1)
                mask_np[channel_idx] = np.maximum(mask_np[channel_idx], np.array(mask_img))

        mask = mask_np # (4, H, W)

        # Apply transformations if any
        if self.transform:
            # Transpose to (H, W, 4) for Albumentations
            mask_hwc = mask.transpose(1, 2, 0)
            augmented = self.transform(image=image_np, mask=mask_hwc)
            image = augmented["image"]
            mask = augmented["mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask.transpose(2, 0, 1)).float() / 255.0
            else:
                mask = mask.permute(2, 0, 1).float()
            
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        else:
            image = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mask).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            image = (image - mean) / std

        # Binarize mask
        mask = (mask > 0.5).float()

        # Compute bbox for the active channel
        x_min, y_min, x_max, y_max = self.compute_bbox(mask[channel_idx])

        # Add random perturbation for data augmentation
        _, h, w = mask.shape
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        noise_w = (
            torch.clamp(torch.randn(1) * bbox_width * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift)
            .round()
            .int()
            .item()
        )
        noise_h = (
            torch.clamp(torch.randn(1) * bbox_height * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift)
            .round()
            .int()
            .item()
        )

        x_min = max(0, x_min + noise_w)
        x_max = min(w, x_max + noise_w)
        y_min = max(0, y_min + noise_h)
        y_max = min(h, y_max + noise_h)
        bboxes = torch.tensor([x_min, y_min, x_max, y_max])

        return image, mask, bboxes

    def compute_bbox(self, mask_tensor):
        """
        Compute the bounding box of the white region in a 2D binary mask tensor.
        """
        if len(mask_tensor.shape) > 2:
            mask_tensor = mask_tensor.squeeze(0)

        rows_any_white = torch.any(mask_tensor == 1, dim=1)
        cols_any_white = torch.any(mask_tensor == 1, dim=0)

        rows_white = torch.where(rows_any_white)[0]
        cols_white = torch.where(cols_any_white)[0]

        if rows_white.nelement() == 0 or cols_white.nelement() == 0:
            return torch.tensor([0, 0, 0, 0])

        y_min, y_max = rows_white[0].item(), rows_white[-1].item()
        x_min, x_max = cols_white[0].item(), cols_white[-1].item()

        return x_min, y_min, x_max, y_max

