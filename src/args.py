import argparse
import os
from pathlib import Path

from src.load_config import DEFAULT_CONFIG_PATH, load_args_from_json


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MobileSAM Training")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path, help="path to the config file")

    # Dataset
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--max_bbox_shift", type=int, help="Maximum bounding box shift")
    parser.add_argument("--num_workers", type=int, help="Number of workers for dataloader")

    # Model
    parser.add_argument("--checkpoint_path", type=str, help="Path to MobileSAM checkpoint")
    parser.add_argument("--model_type", type=str, help="Model type (e.g. vit_t)")
    parser.add_argument("--image_size", type=int, help="Image size")
    parser.add_argument("--save_path", type=str, help="Path to save logs and weights")

    # Training
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--val_freq", type=int, help="Validation frequency")
    parser.add_argument("--gradient_accumulation", type=int, help="Gradient accumulation steps")
    parser.add_argument("--bf16", action="store_true", default=None, help="Use bf16")
    parser.add_argument("--warmup_step", type=int, help="Warmup steps")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume training")

    # Visual
    parser.add_argument("--visual_status", action="store_true", default=None, help="Enable visual status")
    parser.add_argument("--visual_save_path", type=str, help="Path to save visualization")
    parser.add_argument("--IOU_threshold", type=float, help="IOU threshold")

    # Freeze
    parser.add_argument("--freeze_image_encoder", action="store_true", default=None, help="Freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", action="store_true", default=None, help="Freeze prompt encoder")
    parser.add_argument("--freeze_mask_decoder", action="store_true", default=None, help="Freeze mask decoder")

    # HF and Wandb
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, default="", help="Hugging Face Repository ID")
    parser.add_argument(
        "--hf_token", type=str, default="", help="Hugging Face Token (can also be set via HF_TOKEN env var)"
    )
    parser.add_argument("--wandb_project", type=str, default="MobileSAM-finetuning", help="Wandb project name")

    args = parser.parse_args()

    # Load defaults from config
    config = load_args_from_json(args.config)

    # Override with command line arguments if provided
    if args.dataset_dir is not None:
        config.dataset.dataset_dir = args.dataset_dir
    if args.max_bbox_shift is not None:
        config.dataset.max_bbox_shift = args.max_bbox_shift
    if args.num_workers is not None:
        config.dataset.num_workers = args.num_workers

    if args.checkpoint_path is not None:
        config.model.checkpoint_path = args.checkpoint_path
    if args.model_type is not None:
        config.model.type = args.model_type
    if args.image_size is not None:
        config.model.image_size = args.image_size
    if args.save_path is not None:
        config.model.save_path = args.save_path

    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.val_freq is not None:
        config.train.val_freq = args.val_freq
    if args.gradient_accumulation is not None:
        config.train.gradient_accumulation = args.gradient_accumulation
    if args.bf16 is not None:
        config.train.bf16 = args.bf16
    if args.warmup_step is not None:
        config.train.warmup_step = args.warmup_step
    if args.resume is not None:
        config.train.resume = args.resume

    if args.visual_status is not None:
        config.visual.status = args.visual_status
    if args.visual_save_path is not None:
        config.visual.save_path = args.visual_save_path
    if args.IOU_threshold is not None:
        config.visual.IOU_threshold = args.IOU_threshold

    if args.freeze_image_encoder is not None:
        config.freeze.freeze_image_encoder = args.freeze_image_encoder
    if args.freeze_prompt_encoder is not None:
        config.freeze.freeze_prompt_encoder = args.freeze_prompt_encoder
    if args.freeze_mask_decoder is not None:
        config.freeze.freeze_mask_decoder = args.freeze_mask_decoder

    config.push_to_hub = args.push_to_hub
    config.hf_repo_id = args.hf_repo_id
    config.hf_token = args.hf_token if args.hf_token else os.getenv("HF_TOKEN", "")
    config.wandb_project = args.wandb_project

    return config


def parse_amg_args():
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an input image or directory of images, "
            "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
            "as well as pycocotools if saving in RLE format."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Path to the directory where masks will be output. Output will be either a folder "
            "of PNGs per image or a single json with COCO-style masks."
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=("Save masks as COCO RLEs in a single json instead of as a folder of PNGs. Requires pycocotools."),
    )

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than "
            "this value in pixels are removed by postprocessing."
        ),
    )

    return parser.parse_args()


def parse_export_args():
    parser = argparse.ArgumentParser(description="Export the SAM prompt encoder and mask decoder to an ONNX model.")

    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint.")

    parser.add_argument("--output", type=str, required=True, help="The filename to save the ONNX model to.")

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
    )

    parser.add_argument(
        "--return-single-mask",
        action="store_true",
        help=(
            "If true, the exported ONNX model will only return the best mask, "
            "instead of returning multiple masks. For high resolution images "
            "this can improve runtime when upscaling masks is expensive."
        ),
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="The ONNX opset version to use. Must be >=11",
    )

    parser.add_argument(
        "--quantize-out",
        type=str,
        default=None,
        help=(
            "If set, will quantize the model and save it with this name. "
            "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
        ),
    )

    parser.add_argument(
        "--gelu-approximate",
        action="store_true",
        help=(
            "Replace GELU operations with approximations using tanh. Useful "
            "for some runtimes that have slow or unimplemented erf ops, used in GELU."
        ),
    )

    parser.add_argument(
        "--use-stability-score",
        action="store_true",
        help=(
            "Replaces the model's predicted mask quality score with the stability "
            "score calculated on the low resolution masks using an offset of 1.0. "
        ),
    )

    parser.add_argument(
        "--return-extra-metrics",
        action="store_true",
        help=(
            "The model will return five results: (masks, scores, stability_scores, "
            "areas, low_res_logits) instead of the usual three. This can be "
            "significantly slower for high resolution outputs."
        ),
    )

    return parser.parse_args()
