import argparse
import json
import os
from typing import Any

import cv2
import numpy as np

from mobile_sam import SAMAutomaticMaskGenerator, sam_model_registry
from src.args import parse_amg_args
from src.load_checkpoint import get_sam_vit_t

LABEL_MAPS = {
    "coffee": {
        0: "Bệnh sâu vẽ bùa",
        1: "Bệnh phấn trắng",
        2: "Bệnh nấm rỉ sắt",
        3: "Bệnh đốm rong",
    },
    "rice": {
        0: "BrownSpot",
        1: "Healthy",
        2: "Hispa",
        3: "LeafBlast",
    },
}


def write_masks_to_folder(masks: list[dict[str, Any]], path: str, dataset_type: str = "coffee") -> None:

    header = (
        "id,label,label_name,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,"
        "stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"
    )
    metadata = [header]
    label_map = LABEL_MAPS.get(dataset_type, LABEL_MAPS["coffee"])

    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)

        label_id = mask_data.get("label", 0)
        label_name = label_map.get(label_id, str(label_id))

        mask_metadata = [
            str(i),
            str(label_id),
            label_name,
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    if args.model_type == "vit_t":
        print(f"Loading ViT Tiny MobileSAM from {args.checkpoint}...")
        sam = get_sam_vit_t(
            checkpoint_path=args.checkpoint,
            resume=False,
            num_mask_outputs=args.num_classes,
            allow_download=False,
            strict_checkpoint_shapes=True,
        )
    else:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SAMAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)
        print(f"[DEBUG] len(masks) = {len(masks)}")
        print(f"[DEBUG] type(masks) = {type(masks)}")
        print(f"[DEBUG] masks = {masks}")

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            write_masks_to_folder(masks, save_base, args.dataset)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)

        # Visualization
        vis_image = image.copy()
        label_map = LABEL_MAPS.get(args.dataset, LABEL_MAPS["coffee"])
        for mask_data in masks:
            mask = mask_data["segmentation"]
            if output_mode == "coco_rle":
                from mobile_sam.utils.amg import rle_to_mask

                mask = rle_to_mask(mask)

            label_id = mask_data.get("label", 0)
            label_name = label_map.get(label_id, str(label_id))

            np.random.seed(label_id)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = color
            vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.5, 0)

            x, y, _, _ = [int(v) for v in mask_data["bbox"]]
            cv2.putText(vis_image, label_name, (x, max(y - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        vis_path = save_base + "_vis.jpg" if output_mode == "coco_rle" else os.path.join(save_base, "visualization.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print("Done!")


if __name__ == "__main__":
    args = parse_amg_args()
    main(args)
