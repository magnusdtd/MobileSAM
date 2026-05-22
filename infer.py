import os
import subprocess

img_name = "IMG_20190419_103508.jpg"
disease_type = "BrownSpot"
model_type = "vit_t"

input_dir = f"datasets/rice_leaf_disease/{disease_type}"
output_dir = "outputs/samples"

input_path = os.path.join(input_dir, img_name)
checkpoint_path = "outputs/weights/rice_best.pth"

onnx_fp32 = "outputs/weights/rice_mobile_sam.onnx"
onnx_int8 = "outputs/weights/rice_mobile_sam_quantized.onnx"

# 1. Run Standard PyTorch Inference
print("=" * 60)
print("Running PyTorch Checkpoint Inference...")
print("=" * 60)
cmd_pytorch = [
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
subprocess.run(cmd_pytorch, check=True)

# 2. Run Standard ONNX Inference (FP32)
print("\n" + "=" * 60)
print("Running FP32 ONNX Model Inference...")
print("=" * 60)
cmd_onnx_fp32 = [
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
    "--onnx",
    onnx_fp32,
    "--dataset",
    "rice",
    "--device",
    "cpu",
]
subprocess.run(cmd_onnx_fp32, check=True)

# 3. Run Quantized ONNX Inference (INT8)
print("\n" + "=" * 60)
print("Running Quantized INT8 ONNX Model Inference...")
print("=" * 60)
cmd_onnx_int8 = [
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
    "--onnx",
    onnx_int8,
    "--dataset",
    "rice",
    "--device",
    "cpu",
]
subprocess.run(cmd_onnx_int8, check=True)

print("\n" + "=" * 60)
print("All inference runs in infer.py completed successfully!")
print("=" * 60)
