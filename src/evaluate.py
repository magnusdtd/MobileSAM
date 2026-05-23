import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from src.datasets import SAMDataset
from src.load_checkpoint import get_sam_vit_t
from src.load_config import load_args_from_json
from src.loss import DiceLoss, batch_iou
from src.transform import get_transforms
from src.utils import set_seed


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileSAM Model (PyTorch or ONNX)")
    parser.add_argument(
        "--config",
        default="./configs/mobileSAM.json",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.pth or .pt)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to ONNX model (FP32 or INT8 quantized). If not provided, evaluates PyTorch model.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for PyTorch parts (default: cuda if available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (strongly recommend 1 for latency/ONNX runtime compatibility)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of loader workers (default: config value)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory",
    )

    cli_args = parser.parse_args()

    config = load_args_from_json(Path(cli_args.config))

    config.eval_checkpoint = cli_args.checkpoint
    config.eval_onnx = cli_args.onnx
    config.eval_split = cli_args.split
    config.eval_device = cli_args.device
    config.eval_batch_size = cli_args.batch_size

    if cli_args.num_workers is not None:
        config.dataset.num_workers = cli_args.num_workers
    if cli_args.dataset_dir is not None:
        config.dataset.dataset_dir = cli_args.dataset_dir

    return config


def evaluate_model(args):
    set_seed(3407)

    device = torch.device(args.eval_device)
    print("=" * 60)
    print(f"Starting evaluation on '{args.eval_split}' split using device: {device}")
    if args.eval_onnx:
        print(f"ONNX Model Path: {args.eval_onnx}")
    else:
        print(f"PyTorch Checkpoint: {args.eval_checkpoint}")
    print("=" * 60)

    # Initialize transform & dataset
    _, val_transform = get_transforms(image_size=(args.model.image_size, args.model.image_size))
    dataset = SAMDataset(
        root_dir=args.dataset.dataset_dir,
        transform=val_transform,
        max_bbox_shift=args.dataset.max_bbox_shift,
        split=args.eval_split,
    )
    num_mask_outputs = dataset.num_classes

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.dataset.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model = get_sam_vit_t(
        checkpoint_path=args.eval_checkpoint,
        resume=False,
        num_mask_outputs=num_mask_outputs,
        allow_download=False,
        strict_checkpoint_shapes=True,
    )
    model.to(device)
    model.eval()

    ort_session = None
    if args.eval_onnx is not None:
        providers = ["CPUExecutionProvider"]
        if args.eval_device == "cuda" and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider"] + providers
        print(f"Initializing ONNX session with providers: {providers}")
        ort_session = onnxruntime.InferenceSession(args.eval_onnx, providers=providers)

    # Define validation criteria
    criterion_MSE = nn.MSELoss()
    criterion_Dice = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_map_50 = 0.0
    total_map_50_95 = 0.0
    valid_batches = 0

    enc_latencies = []
    dec_latencies = []
    total_latencies = []

    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Evaluating", total=num_batches)

    with torch.no_grad():
        for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
            image, mask, bbox = image.to(device), mask.to(device), bbox.to(device)
            B = image.shape[0]

            if ort_session is not None:
                # ONNX Mode
                # 1. Run image encoder (in PyTorch) and measure its latency
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_enc_start = time.perf_counter()

                image_embeddings = model.image_encoder(image)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_enc_end = time.perf_counter()
                enc_lat_batch = (t_enc_end - t_enc_start) * 1000.0  # ms for batch
                enc_latency = enc_lat_batch / B  # average per-sample

                # 2. Format inputs and run prompt/mask decoder (in ONNX Runtime)
                pred_masks_list = []
                pred_IOUs_list = []
                dec_lat_list = []

                image_embeddings_np = image_embeddings.cpu().numpy()

                for i in range(B):
                    # Bbox coordinates in 1024x1024 scale (already scaled in SAMDataset)
                    x_min, y_min, x_max, y_max = bbox[i].cpu().numpy()
                    point_coords_np = np.array([[[x_min, y_min], [x_max, y_max]]], dtype=np.float32)
                    point_labels_np = np.array([[2.0, 3.0]], dtype=np.float32)
                    mask_input_np = np.zeros((1, 1, 256, 256), dtype=np.float32)
                    has_mask_input_np = np.array([0.0], dtype=np.float32)
                    orig_im_size_np = np.array([1024.0, 1024.0], dtype=np.float32)

                    ort_inputs = {
                        "image_embeddings": image_embeddings_np[i : i + 1],
                        "point_coords": point_coords_np,
                        "point_labels": point_labels_np,
                        "mask_input": mask_input_np,
                        "has_mask_input": has_mask_input_np,
                        "orig_im_size": orig_im_size_np,
                    }

                    t_dec_start = time.perf_counter()
                    ort_outputs = ort_session.run(None, ort_inputs)
                    t_dec_end = time.perf_counter()
                    dec_lat_list.append((t_dec_end - t_dec_start) * 1000.0)

                    # Slice to remove the single mask prediction (index 0) to align with PyTorch multimask output
                    # Shape: (1, 4, 1024, 1024)
                    pred_masks_list.append(torch.tensor(ort_outputs[0][:, 1:, :, :], device=device))
                    # Shape: (1, 4)
                    pred_IOUs_list.append(torch.tensor(ort_outputs[1][:, 1:], device=device))

                pred_mask = torch.cat(pred_masks_list, dim=0)
                pred_IOU = torch.cat(pred_IOUs_list, dim=0)

                dec_latency = sum(dec_lat_list) / B
                total_latency = enc_latency + dec_latency

            else:
                # PyTorch Mode
                # Measure image encoder latency
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_enc_start = time.perf_counter()

                image_embeddings = model.image_encoder(image)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_enc_end = time.perf_counter()
                enc_latency = ((t_enc_end - t_enc_start) * 1000.0) / B

                # Measure prompt/mask decoder latency
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_dec_start = time.perf_counter()

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=bbox,
                    masks=None,
                )

                low_res_masks, pred_IOU = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                )
                pred_mask = model.postprocess_masks(
                    low_res_masks,
                    input_size=[model.image_encoder.img_size, model.image_encoder.img_size],
                    original_size=1024,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_dec_end = time.perf_counter()
                dec_latency = ((t_dec_end - t_dec_start) * 1000.0) / B
                total_latency = enc_latency + dec_latency

            # Record latencies
            enc_latencies.append(enc_latency)
            dec_latencies.append(dec_latency)
            total_latencies.append(total_latency)

            # Compute standard loss and metrics
            iou = batch_iou(torch.sigmoid(pred_mask), mask)
            loss_focal = sigmoid_focal_loss(pred_mask, mask, reduction="mean")
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)
            loss = loss_focal * 20 + loss_dice + loss_mse

            total_loss += loss.item()

            active = mask.sum((2, 3)) > 0
            if active.any():
                active_iou = iou[active]
                batch_miou = active_iou.mean().item()
                batch_dice = (2 * active_iou / (1 + active_iou)).mean().item()
                batch_map_50 = (active_iou >= 0.5).float().mean().item()
                thresholds = torch.arange(0.5, 1.0, 0.05, device=active_iou.device)
                batch_map_50_95 = (active_iou.unsqueeze(1) >= thresholds).float().mean().item()

                total_iou += batch_miou
                total_dice += batch_dice
                total_map_50 += batch_map_50
                total_map_50_95 += batch_map_50_95
                valid_batches += 1

            progress_bar.set_postfix(loss=loss.item(), latency_ms=total_latency)

    # Average metrics over the dataset
    average_loss = total_loss / num_batches
    avg_iou = total_iou / valid_batches if valid_batches > 0 else 0
    avg_dice = total_dice / valid_batches if valid_batches > 0 else 0
    avg_map_50 = total_map_50 / valid_batches if valid_batches > 0 else 0
    avg_map_50_95 = total_map_50_95 / valid_batches if valid_batches > 0 else 0

    mean_enc_latency = np.mean(enc_latencies)
    mean_dec_latency = np.mean(dec_latencies)
    mean_total_latency = np.mean(total_latencies)

    print("\n" + "=" * 60)
    print("                 MobileSAM Evaluation Results")
    print("=" * 60)
    print(f" Model Type:         {'ONNX' if args.eval_onnx else 'PyTorch Checkpoint'}")
    if args.eval_onnx:
        print(f" Model Path:         {args.eval_onnx}")
    else:
        print(f" Checkpoint Path:    {args.eval_checkpoint}")
    print(f" Dataset Split:      {args.eval_split}")
    print(f" Device:             {args.eval_device}")
    print(f" Total Batches:      {num_batches}")
    print(f" Total Samples:      {len(dataset)}")
    print("-" * 60)
    print(f" Loss:               {average_loss:.4f}")
    print(f" mIoU:               {avg_iou:.4f}")
    print(f" Dice Score:         {avg_dice:.4f}")
    print(f" mAP@50:             {avg_map_50:.4f}")
    print(f" mAP@50:95:          {avg_map_50_95:.4f}")
    print("-" * 60)
    print(" Latency Profile (average per sample):")
    print(f"  - Image Encoder:   {mean_enc_latency:.2f} ms")
    print(f"  - Mask Decoder:    {mean_dec_latency:.2f} ms")
    print(f"  - Total Latency:   {mean_total_latency:.2f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_eval_args()
    evaluate_model(args)
