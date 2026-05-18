<div align="center">

# MobileSAM-fast-finetuning

_✨ Finetune MobileSAM with Less Than 4GB RAM!  ✨_

</div>

MobileSAM-fast-finetuning is a training script designed for [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), enabling efficient model finetuning on hardware with limited memory without using adapter.

The script has been tested on both Windows and Linux operating systems:

- Python version: 3.10

- PyTorch version: 2.1

## Installation

1. **PyTorch Installation**: Visit [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) to set up PyTorch on your system.

2. **Dependencies**: Once PyTorch is installed, install the required packages using the command:

```python
pip install -r requirements.txt
```

## Usage

### Preparing the Data

- **Training Data**: Place your training images (JPG format) and corresponding masks (PNG format, same name as the images) in the `./datasets/train` directory.

<p float="center">
  <img src="imgs/dataset_example.png?raw=true" width="70%" />
</p>

- **Validation Data**: Place your validation images (JPG format) and masks (PNG format, same name as the images) in the `./datasets/val` directory.

### Running the Training Script

Run `main.py` to start training. Example command:
```bash
python main.py --batch_size 8 --epochs 10 --push_to_hub --hf_repo_id "<username>/<repo name>"
```

By default, the training status and best checkpoint will be saved at `./outputs/logs/`.
You can customize training settings (like `batch_size`, `epochs`, `learning_rate`, freezing options, etc.) by modifying the JSON configuration file located at `./configs/mobileSAM.json` or by overriding them with command-line arguments:

```bash
# Custom training with configs and arguments
python main.py --config configs/mobileSAM.json --batch_size 4 --epochs 1000 --resume
```

### Running Automatic Mask Generation

To automatically generate binary masks (PNG format) or COCO-style RLE masks for an image or directory of images, use the automatic mask generation script:

```bash
# Basic run generating binary mask folders:
python src/amg.py --input path/to/images --output path/to/output_dir --model-type vit_t --checkpoint outputs/weights/mobile_sam.pt

# Generate masks and convert them to COCO RLE JSON:
python src/amg.py --input path/to/images --output path/to/output_dir --model-type vit_t --checkpoint outputs/weights/mobile_sam.pt --convert-to-rle
```

### Exporting the Model to ONNX

You can export the fine-tuned model's prompt encoder and mask decoder to an ONNX model (and optionally dynamically quantize it to INT8 format for optimized execution):

```bash
# Standard ONNX export:
python src/export_onnx_model.py --checkpoint outputs/weights/mobile_sam.pt --output mobile_sam.onnx --model-type vit_t

# Export with dynamic quantization:
python src/export_onnx_model.py --checkpoint outputs/weights/mobile_sam.pt --output mobile_sam.onnx --quantize-out mobile_sam_quantized.onnx --model-type vit_t
```

### Visualizing Annotations

To visualize overlaid mask annotations (COCO-style JSON files) on a target image:

```bash
python scripts/show_masks.py --image_path path/to/image.jpg --annotation_path path/to/annotation.json
```

## Inference

To use the finetuned MobileSAM model, simply replace the original MobileSAM checkpoint with the newly finetuned one. No additional configuration is needed for a seamless transition!

## To do list
- [x] Unify the argument parser
- [x] Replace TensorBoard with Wandb
- [x] Resume checkpoint training from the last finetuned checkpoint
- [x] Use Albumentations to augment the data
- [x] Add an option to push the model to HF


## References
- [MobileSAM-fast-finetuning](https://github.com/KdaiP/MobileSAM-fast-finetuning)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [Medical-SAM-Adapter](https://github.com/WuJunde/Medical-SAM-Adapter)
- [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)