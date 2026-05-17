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

Run `main.py`. Example command:
```bash
python main.py --batch_size 8 --epochs 1 --push_to_hub --hf_repo_id "<username>/<repo name>"
```

By default, the checkpoint will be saved at  `./outputs/logs/`

To customize your training settings, such as batch_size, you can modify the configuration file located at `./configs/mobileSAM.json`. Please note that for users of PyTorch versions earlier than 2.1, enabling `bf16=true` in the configuration may result in errors.

## Inference

To use the finetuned MobileSAM model, simply replace the original MobileSAM checkpoint with the newly finetuned one. No additional configuration needed for a seamless transition!

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