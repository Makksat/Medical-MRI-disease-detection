from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from src.utils import ensure_dir, save_history_csv, save_json


def train_one_epoch(
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
        labels = batch["label"].float().to(device).view(-1, 1)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_classification(
    model: torch.nn.Module,
    loader,
    loss_fn,
    device: torch.device,
) -> dict:
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device).view(-1, 1)

        logits = model(images)
        loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).float()

        running_loss += loss.item()
        all_probs.extend(probs.cpu().numpy().ravel().tolist())
        all_preds.extend(preds.cpu().numpy().ravel().tolist())
        all_labels.extend(labels.cpu().numpy().ravel().tolist())

    val_loss = running_loss / max(len(loader), 1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    acc = accuracy_score(all_labels, all_preds)

    return {
        "val_loss": val_loss,
        "val_auc": auc,
        "val_acc": acc,
    }


def fit_classification_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epochs: int,
    out_dir: str,
    experiment_name: str = "classification_baseline",
) -> tuple[list[dict], str]:
    ensure_dir(out_dir)

    best_auc = -1.0
    best_model_path = f"{out_dir}/{experiment_name}_best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        metrics = validate_classification(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "val_auc": metrics["val_auc"],
            "val_acc": metrics["val_acc"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_auc={metrics['val_auc']:.4f} "
            f"val_acc={metrics['val_acc']:.4f}"
        )

        if metrics["val_auc"] > best_auc:
            best_auc = metrics["val_auc"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    save_history_csv(history, f"{out_dir}/{experiment_name}_history.csv")
    save_json(
        {
            "experiment_name": experiment_name,
            "best_val_auc": best_auc,
            "epochs": epochs,
        },
        f"{out_dir}/{experiment_name}_summary.json",
    )

    return history, best_model_path
