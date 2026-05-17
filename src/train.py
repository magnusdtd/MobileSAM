import torch
import wandb
from torch.cuda.amp import autocast
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from src.loss import batch_iou
from src.transform import MEAN, STD
from src.visualization import overlay_mask_on_image


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
        with autocast(enabled=args.train.bf16, dtype=torch.bfloat16):
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(mask, torch.sigmoid(pred_mask))

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
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Validating", total=num_batches)

    # Evaluation mode: no gradients needed
    with torch.no_grad():
        for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
            # Move input and target data to the GPU
            image, mask, bbox = image.cuda(), mask.cuda(), bbox.cuda()

            # Forward pass
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(mask, torch.sigmoid(pred_mask))

            loss_focal = sigmoid_focal_loss(pred_mask, mask, reduction="mean")
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)
            loss = loss_focal * 20 + loss_dice + loss_mse

            # Accumulate the loss for logging
            total_loss += loss.item()
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

    # Log the validation loss to Wandb
    wandb.log({"Val loss": average_loss, "epoch": epoch})

    return average_loss
