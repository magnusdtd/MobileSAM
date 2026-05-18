from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from huggingface_hub import HfApi
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.args import parse_args
from src.datasets import SAMDataset
from src.load_checkpoint import get_sam_vit_t
from src.load_logger import Logger
from src.loss import DiceLoss
from src.schedular import LinearWarmup
from src.train import train_epoch, val_epoch
from src.transform import get_transforms
from src.utils import save_checkpoint, set_seed

set_seed(3407)


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

    # Initialize the best validation loss variable
    best_val_loss = float("inf")
    start_epoch = 0

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
            val_loss = val_epoch(args, val_loader, model, criterion_MSE, criterion_Dice, epoch, scaler)
            logger.info(f"Epoch {epoch + 1}/{args.train.epochs}, Val Loss: {val_loss:.4f}")

            # Save the best model based on validation loss
            # the best model could be used like the original MobileSAM checkpoint without any modification
            is_best = val_loss < best_val_loss
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, is_best, save_path
            )
            if is_best:
                best_val_loss = val_loss

    # Run inference on the test set
    logger.info("Running evaluation on Test Set...")
    best_model_path = save_path / "best.pth"
    if best_model_path.exists():
        ckpt = torch.load(best_model_path)
        model.load_state_dict(ckpt["model"])

    test_loss = val_epoch(args, test_loader, model, criterion_MSE, criterion_Dice, args.train.epochs, scaler)
    logger.info(f"Final Test Loss: {test_loss:.4f}")

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
