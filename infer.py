import os
import subprocess

img_name = "IMG_20190421_200148"
disease_type = "BrownSpot"
model_type = "vit_t"

input_dir = f"datasets/rice_leaf_disease/{disease_type}"
output_dir = "outputs/samples"

input_path = os.path.join(input_dir, f"{img_name}.jpg")
checkpoint_path = "outputs/weights/best.pth"
vis_path = os.path.join(output_dir, img_name, "visualization.jpg")


command = [
    "uv",
    "run",
    "-m",
    "src.amg",
    "--input",
    input_path,
    "--output",
    output_dir,
    "--model-type",
    model_type,
    "--checkpoint",
    checkpoint_path,
    "--device",
    "cpu",
    "--pred-iou-thresh",
    "0.5",
    "--stability-score-thresh",
    "0.5",
]

subprocess.run(command, check=True)
