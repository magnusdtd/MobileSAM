
# MobileSAM Fine-tuning

This project is a training script designed for [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), enabling efficient model finetuning on hardware with limited memory without using adapters.

This project is configured to use [uv](https://github.com/astral-sh/uv) for fast and reliable Python package and environment management. The dataset uses the **COCO format** for annotations, and the codebase is designed to train 2 models for 2 specific datasets: **Rice** and **Coffee** leaf disease.

---

## 🛠️ Installation & Setup

This project uses `uv` to manage its virtual environment and dependencies.

1. **Install uv**: Follow the [official uv installation guide](https://github.com/astral-sh/uv) to set up `uv` on your machine.
2. **Synchronize Dependencies**: Initialize the environment and install all package requirements automatically:
   ```bash
   uv sync
   ```
3. **Download Base Checkpoints**: Download the pre-trained weights from Hugging Face Hub:
   ```bash
   uv run download_ckpt.py
   ```

---

## 📁 Dataset Structure

The datasets use the **COCO JSON format** for annotations. The images and their corresponding JSON files should be placed inside the `datasets/` directory:

```text
datasets/
├── coffee_leaf_disease/
│   ├── annotations.coco.json
│   ├── 0/
│   ├── 1/
│   └── ... (other image folders)
└── rice_leaf_disease/
    ├── annotations.coco.json
    ├── BrownSpot/
    ├── Healthy/
    └── ... (other image folders)
```

### Automatic Splitting
The custom dataset class `SAMDataset` at `src/datasets.py` loads the images and `annotations.coco.json` files, then splits the data into a **70% Train / 10% Validation / 20% Test** split deterministically using a fixed random seed.

---

## 🚀 Running the Training Script

You can train two separate models tailored to the respective datasets. To do this, specify the appropriate `--dataset_dir` and add a `--checkpoint_prefix` to easily identify your saved weights.

### 1. Train Rice Leaf Disease Model
```bash
uv run python src/train.py \
  --dataset_dir datasets/rice_leaf_disease \
  --checkpoint_prefix rice_ \
  --epochs 100 \
  --batch_size 8
```

### 2. Train Coffee Leaf Disease Model
```bash
uv run python src/train.py \
  --dataset_dir datasets/coffee_leaf_disease \
  --checkpoint_prefix coffee_ \
  --epochs 100 \
  --batch_size 8
```

> [!NOTE]
> By default, the training status and best checkpoint will be saved in `./outputs/logs/` (e.g. `rice_best.pth` and `coffee_best.pth`).
> You can customize training parameters (such as learning rate, batch size, epochs, and frozen layers) by editing `configs/mobileSAM.json` or by overriding them with command-line arguments:
> ```bash
> uv run python src/train.py --config configs/mobileSAM.json --batch_size 4 --epochs 1000 --resume
> ```

---

## 🔍 Inference & Testing

To test model inference on a sample image or dataset:

```bash
uv run python infer.py
```

To run the automatic mask generation (AMG) script directly:
```bash
# Basic run generating binary mask folders:
uv run python src/amg.py \
  --input path/to/images \
  --output path/to/output_dir \
  --model-type vit_t \
  --checkpoint outputs/weights/rice_best.pth

# Generate masks and convert them to COCO RLE JSON:
uv run python src/amg.py \
  --input path/to/images \
  --output path/to/output_dir \
  --model-type vit_t \
  --checkpoint outputs/weights/rice_best.pth \
  --convert-to-rle
```

---

## 📦 Exporting the Model to ONNX

Export the fine-tuned model's prompt encoder and mask decoder to an ONNX model (and optionally dynamically quantize it to INT8 format for optimized execution):

```bash
# Standard ONNX export:
uv run python src/export_onnx_model.py \
  --checkpoint outputs/weights/rice_best.pth \
  --output rice_sam.onnx \
  --model-type vit_t

# Export with dynamic quantization:
uv run python src/export_onnx_model.py \
  --checkpoint outputs/weights/rice_best.pth \
  --output rice_sam.onnx \
  --quantize-out rice_sam_quantized.onnx \
  --model-type vit_t
```

---

## 📊 Visualizing Annotations

To visualize overlaid mask annotations (COCO-style JSON files) on a target image:

```bash
uv run python scripts/show_masks.py \
  --image_path path/to/image.jpg \
  --annotation_path path/to/annotation.json
```

---

## Model comparison

| Model Type | mAP@50 | mAP@50:95 | mIoU | Dice | Inference (ms) | Size (MB) |
|---------|--------|-----------|------|------|----------------|-----------|
| Rice MobileSAM (PyTorch) | 0.5637 | 0.5141 | 0.5440 | 0.5712 | 65.90 | 41.3 |
| Rice MobileSAM (ONNX) | 0.5637 | 0.5141 | 0.5440 | 0.5712 | 160.24 | 17.1 |
| Rice MobileSAM (ONNX Quantized) | 0.5469 | 0.5008 | 0.5315 | 0.5589 | 166.07 | 8.96 |
| Coffee MobileSAM (PyTorch) | 0.6923 | 0.6591 | 0.6671 | 0.6848 | 62.14 | 41.3 |
| Coffee MobileSAM (ONNX) | 0.6923 | 0.6591 | 0.6671 | 0.6848 | 146.31 | 17.1 |
| Coffee MobileSAM (ONNX Quantized) | 0.6923 | 0.6559 | 0.6639 | 0.6820 | 148.13 | 8.96 |

---

## 📝 To-Do List

- [x] Unify the argument parser
- [x] Replace TensorBoard with Wandb
- [x] Resume checkpoint training from the last finetuned checkpoint
- [x] Use Albumentations to augment the data
- [x] Add an option to push the model to HF
- [x] Add mAP@50, mAP@50:95, mIoU, Dice Score in the validation phase and make a log for them
- [x] Early stopping with mAP@50
- [x] Add prefix (`rice_` and `coffee_`) to checkpoints
- [x] Test and fix the inference script
- [x] Test and fix the ONNX export script
- [x] Add evaluating scripts to calculate the performance of the model in Pytorch and ONNX format
- [ ] Hyperparameter tuning

---

## 📚 References

- [MobileSAM-fast-finetuning](https://github.com/KdaiP/MobileSAM-fast-finetuning)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [Medical-SAM-Adapter](https://github.com/WuJunde/Medical-SAM-Adapter)
- [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)
