from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from huggingface_hub import HfApi
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from src.args import parse_args
from src.datasets import SAMDataset
from src.load_checkpoint import get_sam_vit_t
from src.load_logger import Logger
from src.loss import DiceLoss, batch_iou
from src.schedular import LinearWarmup
from src.transform import MEAN, STD, get_transforms
from src.utils import save_checkpoint, set_seed
from src.visualization import overlay_mask_on_image

set_seed(3407)


def train_epoch(
    args,
    dataloader,
    model,
    optimizer,
    criterion_MSE,
    criterion_Dice,
    epoch,
    scaler,
    lr_scheduler,
    warmup_scheduler,
):
    """Main training function."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Training", total=num_batches)

    for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
        # Move input and target data to the GPU
        image, mask, bbox = image.cuda(non_blocking=True), mask.cuda(non_blocking=True), bbox.cuda(non_blocking=True)

        # Forward pass with mixed precision
        with autocast(device_type="cuda", enabled=args.train.bf16, dtype=torch.bfloat16):
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(torch.sigmoid(pred_mask), mask)

            loss_focal = sigmoid_focal_loss(pred_mask, mask, reduction="mean")
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)
            loss = loss_focal * 20 + loss_dice + loss_mse

        # Backward pass and update model parameters
        scaler.scale(loss).backward()
        if batch_idx % args.train.gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        # Accumulate the loss for logging
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Calculate average training loss for the epoch
    average_loss = total_loss / num_batches

    # Log the training loss to Wandb
    log_dict = {"Training loss": average_loss, "epoch": epoch}
    for i, param_group in enumerate(optimizer.param_groups):
        log_dict[f"Learning_rate/group_{i}"] = param_group["lr"]
    wandb.log(log_dict)

    return average_loss


def val_epoch(
    args,
    dataloader,
    model,
    criterion_MSE,
    criterion_Dice,
    epoch,
    scaler,
):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_map_50 = 0.0
    total_map_50_95 = 0.0
    valid_batches = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Validating", total=num_batches)

    # Evaluation mode: no gradients needed
    with torch.no_grad():
        for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
            # Move input and target data to the GPU
            image, mask, bbox = image.cuda(), mask.cuda(), bbox.cuda()

            # Forward pass
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(torch.sigmoid(pred_mask), mask)

            loss_focal = sigmoid_focal_loss(pred_mask, mask, reduction="mean")
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)
            loss = loss_focal * 20 + loss_dice + loss_mse

            # Accumulate the loss for logging
            total_loss += loss.item()

            # Calculate metrics for active channels
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

            progress_bar.set_postfix(loss=loss.item())

            if args.visual.status:
                vis_image = image[0]
                vis_mask = pred_mask[0]
                vis_bbox = bbox[0]
                vis_mask = torch.sigmoid(vis_mask)
                mean = torch.tensor(MEAN, device=vis_image.device)
                std = torch.tensor(STD, device=vis_image.device)
                vis_image = vis_image * std[:, None, None] + mean[:, None, None]
                overlay_mask_on_image(
                    vis_image,
                    vis_mask,
                    vis_bbox,
                    threshold=args.visual.IOU_threshold,
                    save_dir=args.visual.save_path,
                    info=(epoch, batch_idx),
                )

        # Calculate average validation loss for the epoch
        average_loss = total_loss / num_batches
        avg_iou = total_iou / valid_batches if valid_batches > 0 else 0
        avg_dice = total_dice / valid_batches if valid_batches > 0 else 0
        avg_map_50 = total_map_50 / valid_batches if valid_batches > 0 else 0
        avg_map_50_95 = total_map_50_95 / valid_batches if valid_batches > 0 else 0

    # Log the validation loss to Wandb
    wandb.log(
        {
            "Val loss": average_loss,
            "mIoU": avg_iou,
            "Dice Score": avg_dice,
            "mAP@50": avg_map_50,
            "mAP@50:95": avg_map_50_95,
            "epoch": epoch,
        }
    )

    return average_loss, avg_map_50


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available."

    train_transform, val_transform = get_transforms(image_size=(args.model.image_size, args.model.image_size))

    train_dataset = SAMDataset(
        root_dir=args.dataset.dataset_dir,
        transform=train_transform,
        max_bbox_shift=args.dataset.max_bbox_shift,
        split="train",
    )
    val_dataset = SAMDataset(
        root_dir=args.dataset.dataset_dir,
        transform=val_transform,
        max_bbox_shift=args.dataset.max_bbox_shift,
        split="val",
    )
    test_dataset = SAMDataset(
        root_dir=args.dataset.dataset_dir,
        transform=val_transform,
        max_bbox_shift=args.dataset.max_bbox_shift,
        split="test",
    )
    num_mask_outputs = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train.batch_size,
        num_workers=args.dataset.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train.batch_size,
        num_workers=args.dataset.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.train.batch_size,
        num_workers=args.dataset.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    # Define checkpoint and saving paths
    checkpoint_path = Path(args.model.checkpoint_path)
    save_path = Path(args.model.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize the logger
    logger = Logger(save_path / "training.log").get_logger()

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Load the MobileSAM checkpoint and move it to CUDA
    # get_sam_vit_t handles base weights
    model = get_sam_vit_t(
        checkpoint_path=checkpoint_path,
        resume=False,
        num_mask_outputs=num_mask_outputs,
    ).cuda()

    # Conditionally freeze layers based on args
    for param in model.image_encoder.parameters():
        param.requires_grad = not args.freeze.freeze_image_encoder
    for param in model.prompt_encoder.parameters():
        param.requires_grad = not args.freeze.freeze_prompt_encoder
    for param in model.mask_decoder.parameters():
        param.requires_grad = not args.freeze.freeze_mask_decoder

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.epochs * len(train_dataset))
    warmup_scheduler = LinearWarmup(optimizer, warmup_period=args.train.warmup_step)

    criterion_MSE = nn.MSELoss()
    criterion_Dice = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    # Initialize the best validation map_50 variable
    best_map_50 = 0.0
    start_epoch = 0
    epochs_no_improve = 0
    patience = getattr(args.train, "patience", 10)

    # Resume from checkpoint if specified
    if args.train.resume:
        last_ckpt_path = save_path / "last.pth"
        if last_ckpt_path.exists():
            logger.info(f"Resuming training from {last_ckpt_path}")
            ckpt = torch.load(last_ckpt_path)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1

    # Initialize Wandb for logging
    wandb.init(project=getattr(args, "wandb_project", "MobileSAM-finetuning"), config=vars(args))

    # Main training loop
    for epoch in range(start_epoch, args.train.epochs):
        # Train for one epoch
        train_loss = train_epoch(
            args,
            train_loader,
            model,
            optimizer,
            criterion_MSE,
            criterion_Dice,
            epoch,
            scaler,
            lr_scheduler,
            warmup_scheduler,
        )
        logger.info(f"Epoch {epoch + 1}/{args.train.epochs}, Train Loss: {train_loss:.4f}")

        # Validate and save the model at specified intervals
        if (epoch + 1) % args.train.val_freq == 0:
            val_loss, map_50 = val_epoch(args, val_loader, model, criterion_MSE, criterion_Dice, epoch, scaler)
            logger.info(f"Epoch {epoch + 1}/{args.train.epochs}, Val Loss: {val_loss:.4f}, mAP@50: {map_50:.4f}")

            # Save the best model based on validation mAP@50
            # the best model could be used like the original MobileSAM checkpoint without any modification
            is_best = map_50 > best_map_50
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, is_best, save_path
            )
            if is_best:
                best_map_50 = map_50
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs. Best mAP@50: {best_map_50:.4f}")
                    break

    # Run inference on the test set
    logger.info("Running evaluation on Test Set...")
    best_model_path = save_path / "best.pth"
    if best_model_path.exists():
        ckpt = torch.load(best_model_path)
        model.load_state_dict(ckpt)

    test_loss, test_map_50 = val_epoch(
        args, test_loader, model, criterion_MSE, criterion_Dice, args.train.epochs, scaler
    )
    logger.info(f"Final Test Loss: {test_loss:.4f}, Test mAP@50: {test_map_50:.4f}")

    # Push to Hugging Face Hub if configured
    if getattr(args, "push_to_hub", False) and getattr(args, "hf_repo_id", ""):
        logger.info(f"Pushing best model to Hugging Face Hub: {args.hf_repo_id}")
        api = HfApi()
        best_model_path = save_path / "best.pth"
        if best_model_path.exists():
            api.upload_file(
                path_or_fileobj=str(best_model_path),
                path_in_repo="best.pth",
                repo_id=args.hf_repo_id,
                token=args.hf_token,
                commit_message="Upload fine-tuned MobileSAM model",
            )
            logger.info("Model pushed successfully.")
        else:
            logger.error("Best model checkpoint not found for upload.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
