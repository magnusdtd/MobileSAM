import logging
from pathlib import Path
from typing import Any

import requests
import torch
from torch import nn
from torch.nn import functional as F

from mobile_sam.modeling import TwoWayTransformer
from mobile_sam.modeling.image_encoder import ImageEncoderViT
from mobile_sam.modeling.mask_decoder import MaskDecoder
from mobile_sam.modeling.prompt_encoder import PromptEncoder
from mobile_sam.modeling.tiny_vit_sam import TinyViT
from src.load_logger import Logger


class SAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT | TinyViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: list[float] = [123.675, 116.28, 103.53],
        pixel_std: list[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, image, bbox):
        # input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(image)

        points = None
        boxes = bbox
        masks = None

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=[self.image_encoder.img_size, self.image_encoder.img_size],
            original_size=1024,
        )
        # masks = masks > self.mask_threshold

        return masks, iou_predictions

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


def _load_matching_state_dict(
    model: nn.Module,
    checkpoint_path: Path,
    strict_shapes: bool = False,
) -> None:
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]

    model_state = model.state_dict()

    # Special handling: Check if checkpoint has 6 MLPs and model expects 5 (discarding 'uit' class)
    checkpoint_has_6_mlps = any(key.startswith("mask_decoder.output_hypernetworks_mlps.5.") for key in state_dict)
    model_has_5_mlps = (
        "mask_decoder.output_hypernetworks_mlps.4.layers.0.weight" in model_state
        and "mask_decoder.output_hypernetworks_mlps.5.layers.0.weight" not in model_state
    )

    if checkpoint_has_6_mlps and model_has_5_mlps:
        logging.info(
            "Checkpoint has 6 output MLPs but model expects 5. Renaming checkpoint keys to skip class 'uit' (index 1)..."
        )
        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("mask_decoder.output_hypernetworks_mlps."):
                parts = key.split(".")
                idx = int(parts[2])
                if idx == 0:
                    mapped_state_dict[key] = value
                elif idx == 1:
                    # Skip 'uit' MLP
                    continue
                else:
                    parts[2] = str(idx - 1)
                    new_key = ".".join(parts)
                    mapped_state_dict[new_key] = value
            else:
                mapped_state_dict[key] = value
        state_dict = mapped_state_dict

    compatible_state = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                # Custom handling for shape mismatched mask decoder parameters
                if key == "mask_decoder.mask_tokens.weight":
                    new_weight = model_state[key].clone()
                    if value.shape[0] == 6 and new_weight.shape[0] == 5:
                        new_weight[0] = value[0]
                        new_weight[1:] = value[2:]
                        compatible_state[key] = new_weight
                        logging.info(f"Custom loaded {key}: mapped from 6 to 5 tokens (skipping index 1)")
                    else:
                        min_tokens = min(value.shape[0], new_weight.shape[0])
                        new_weight[:min_tokens] = value[:min_tokens]
                        if new_weight.shape[0] > value.shape[0]:
                            # Fill the remaining slots with the single_mask token (index 0)
                            new_weight[min_tokens:] = value[0]
                        compatible_state[key] = new_weight
                        logging.info(f"Custom loaded {key}: expanded from {value.shape} to {new_weight.shape}")
                elif key == "mask_decoder.iou_prediction_head.layers.2.weight":
                    new_weight = model_state[key].clone()
                    if value.shape[0] == 6 and new_weight.shape[0] == 5:
                        new_weight[0] = value[0]
                        new_weight[1:] = value[2:]
                        compatible_state[key] = new_weight
                        logging.info(f"Custom loaded {key}: mapped from 6 to 5 units (skipping index 1)")
                    else:
                        min_tokens = min(value.shape[0], new_weight.shape[0])
                        new_weight[:min_tokens] = value[:min_tokens]
                        if new_weight.shape[0] > value.shape[0]:
                            new_weight[min_tokens:] = value[0]
                        compatible_state[key] = new_weight
                        logging.info(f"Custom loaded {key}: expanded from {value.shape} to {new_weight.shape}")
                elif key == "mask_decoder.iou_prediction_head.layers.2.bias":
                    new_bias = model_state[key].clone()
                    if value.shape[0] == 6 and new_bias.shape[0] == 5:
                        new_bias[0] = value[0]
                        new_bias[1:] = value[2:]
                        compatible_state[key] = new_bias
                        logging.info(f"Custom loaded {key}: mapped from 6 to 5 units (skipping index 1)")
                    else:
                        min_tokens = min(value.shape[0], new_bias.shape[0])
                        new_bias[:min_tokens] = value[:min_tokens]
                        if new_bias.shape[0] > value.shape[0]:
                            new_bias[min_tokens:] = value[0]
                        compatible_state[key] = new_bias
                        logging.info(f"Custom loaded {key}: expanded from {value.shape} to {new_bias.shape}")
                else:
                    skipped_keys.append(key)
        else:
            skipped_keys.append(key)

    if strict_shapes and skipped_keys:
        details = "\n".join(
            f"- {key}: checkpoint {tuple(value.shape) if hasattr(value, 'shape') else type(value)}, "
            f"model {tuple(model_state[key].shape) if key in model_state else 'missing'}"
            for key, value in state_dict.items()
            if key not in compatible_state
        )
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with this model configuration.\n"
            f"Requested num_mask_outputs={model.mask_decoder.num_multimask_outputs}, "
            f"which expects {model.mask_decoder.num_mask_tokens} mask tokens including the single-mask token.\n"
            f"Incompatible keys:\n{details}"
        )

    model.load_state_dict(compatible_state, strict=False)

    # If the model has extra mask output MLPs (checkpoint has 4 MLPs corresponding to 4 mask tokens),
    # copy the weights of MLP 0 (single_mask) to the extra MLPs.
    if hasattr(model, "mask_decoder") and hasattr(model.mask_decoder, "num_mask_tokens"):
        num_mask_tokens = model.mask_decoder.num_mask_tokens
        checkpoint_mlps = 4
        if num_mask_tokens > checkpoint_mlps:
            for i in range(checkpoint_mlps, num_mask_tokens):
                model.mask_decoder.output_hypernetworks_mlps[i].load_state_dict(
                    model.mask_decoder.output_hypernetworks_mlps[0].state_dict()
                )
            logging.info(
                f"Copied weights from output_hypernetworks_mlps[0] to index {checkpoint_mlps} through {num_mask_tokens - 1}"
            )

    if skipped_keys:
        logging.info("Skipped %d checkpoint keys with incompatible shapes.", len(skipped_keys))


def get_sam_vit_t(
    checkpoint_path=None,
    resume=False,
    num_mask_outputs=3,
    allow_download=True,
    strict_checkpoint_shapes=False,
):

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = SAM(
        image_encoder=TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_mask_outputs,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint_path is not None and resume is False:
        if not checkpoint_path.is_file():
            if not allow_download:
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}. "
                    "Train saves the best checkpoint under model.save_path, "
                    "which defaults to outputs/logs/best.pth."
                )
            logger = Logger(checkpoint_path.parent / "log.log").get_logger()
            logger.info(f"Downloading MobileSAM checkpoint to {checkpoint_path}...")
            url = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
            filename = "mobile_sam.pt"

            logger.info(f"Downloading {filename}...")

            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(checkpoint_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            print("Download complete!")

        _load_matching_state_dict(mobile_sam, checkpoint_path, strict_shapes=strict_checkpoint_shapes)
        logging.info(f"Using pretrained model: {checkpoint_path}")
    return mobile_sam
