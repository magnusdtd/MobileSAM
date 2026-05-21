import os
import subprocess

img_name = "IMG_20190419_094238"
disease_type = "Hispa"
model_type = "vit_t"

input_dir = f"datasets/rice_leaf_disease/{disease_type}"
output_dir = "outputs/samples"

input_path = os.path.join(input_dir, f"{img_name}.jpg")
checkpoint_path = "outputs/weights/rice_best.pth"
vis_path = os.path.join(output_dir, img_name, "visualization.jpg")


command = [
    "uv",
    "run",
    "-m",
    "src.infer_bbox",
    "--input",
    input_path,
    "--output",
    output_dir,
    "--checkpoint",
    checkpoint_path,
    "--dataset",
    "rice",
    "--device",
    "cpu",
]

subprocess.run(command, check=True)
