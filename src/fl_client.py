from __future__ import annotations

from collections import OrderedDict
from typing import Any

import flwr as fl
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset

from src.models import build_classification_model
from src.train_classification import train_one_epoch, validate_classification
from src.transforms import (
    get_classification_train_transforms,
    get_classification_val_transforms,
)
from src.utils import get_device


def get_model_parameters(model: torch.nn.Module) -> list:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: list) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class MRIClassificationClient(fl.client.NumPyClient):
    def __init__(
        self,
        train_samples: list[dict],
        val_samples: list[dict],
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        local_epochs: int = 1,
        image_size: tuple[int, int, int] = (96, 96, 96),
    ) -> None:
        self.device = get_device()
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        self.model = build_classification_model(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
        ).to(self.device)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        train_ds = Dataset(
            data=train_samples,
            transform=get_classification_train_transforms(image_size=image_size),
        )
        val_ds = Dataset(
            data=val_samples,
            transform=get_classification_val_transforms(image_size=image_size),
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

    def get_parameters(self, config: dict[str, Any]):
        return get_model_parameters(self.model)

    def fit(self, parameters, config: dict[str, Any]):
        set_model_parameters(self.model, parameters)

        for _ in range(self.local_epochs):
            train_loss = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                device=self.device,
            )

        return get_model_parameters(self.model), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
        }

    def evaluate(self, parameters, config: dict[str, Any]):
        set_model_parameters(self.model, parameters)

        metrics = validate_classification(
            model=self.model,
            loader=self.val_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )

        return float(metrics["val_loss"]), len(self.val_loader.dataset), {
            "val_auc": float(metrics["val_auc"]),
            "val_acc": float(metrics["val_acc"]),
        }
