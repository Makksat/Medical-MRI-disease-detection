from __future__ import annotations

import os

import torch
from monai.data import decollate_batch
from tqdm import tqdm

from src.utils import ensure_dir, save_history_csv, save_json


def train_one_segmentation_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_segmentation(
    model: torch.nn.Module,
    loader,
    loss_fn,
    dice_metric,
    post_pred,
    post_label,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0

    dice_metric.reset()

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        outputs_list = decollate_batch(outputs)
        labels_list = decollate_batch(labels)

        outputs_convert = [post_pred(x) for x in outputs_list]
        labels_convert = [post_label(x) for x in labels_list]

        dice_metric(y_pred=outputs_convert, y=labels_convert)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    val_loss = running_loss / max(len(loader), 1)

    return {
        "val_loss": val_loss,
        "val_dice": mean_dice,
    }


def fit_segmentation_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    dice_metric,
    post_pred,
    post_label,
    device: torch.device,
    epochs: int,
    out_dir: str,
    experiment_name: str = "msseg_baseline",
) -> tuple[list[dict], str]:
    ensure_dir(out_dir)

    best_dice = -1.0
    best_epoch = 0
    history = []

    best_model_path = os.path.join(out_dir, f"{experiment_name}_best.pt")
    history_csv_path = os.path.join(out_dir, f"{experiment_name}_history.csv")
    summary_json_path = os.path.join(out_dir, f"{experiment_name}_summary.json")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_segmentation_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        metrics = validate_segmentation(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            dice_metric=dice_metric,
            post_pred=post_pred,
            post_label=post_label,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "val_dice": metrics["val_dice"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_dice={metrics['val_dice']:.4f}"
        )

        if metrics["val_dice"] > best_dice:
            best_dice = metrics["val_dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    save_history_csv(history, history_csv_path)
    save_json(
        {
            "experiment_name": experiment_name,
            "best_val_dice": best_dice,
            "best_epoch": best_epoch,
            "epochs": epochs,
        },
        summary_json_path,
    )

    return history, best_model_path
