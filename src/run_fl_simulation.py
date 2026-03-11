from __future__ import annotations

from pathlib import Path
import glob
import json
import os

import pandas as pd
import flwr as fl
import ray

from src.data import split_classification_data
from src.fl_client import MRIClassificationClient
from src.fl_server import build_fedavg_strategy
from src.fl_utils import print_client_summary, split_samples_iid
from src.utils import set_seed


def load_parkinson_samples(data_root: str) -> list[dict]:
    data_root = Path(data_root)
    participants_tsv = data_root / "participants.tsv"

    if not participants_tsv.exists():
        raise FileNotFoundError(f"Missing participants.tsv: {participants_tsv}")

    df = pd.read_csv(participants_tsv, sep="\t")

    if "participant_id" not in df.columns or "group" not in df.columns:
        raise ValueError(
            "participants.tsv must contain at least 'participant_id' and 'group' columns"
        )

    df["group"] = df["group"].astype(str).str.strip()

    def make_label(g):
        if g.lower() == "control":
            return 0
        if g.upper().startswith("PD"):
            return 1
        return None

    df["label"] = df["group"].apply(make_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    samples = []
    missing = 0

    for _, row in df.iterrows():
        sub = row["participant_id"]
        label = int(row["label"])

        candidates = sorted(
            glob.glob(str(data_root / sub / "anat" / f"{sub}_T1w.nii.gz"))
        )

        if len(candidates) == 0:
            missing += 1
            continue

        samples.append(
            {
                "image": candidates[0],
                "label": label,
                "subject": sub,
                "group": row["group"],
                "age": row["age"] if "age" in df.columns else None,
                "sex": row["sex"] if "sex" in df.columns else None,
            }
        )

    print(f"Loaded {len(samples)} Parkinson samples | missing T1w: {missing}")

    if len(samples) == 0:
        raise ValueError(
            f"No usable T1w samples found in {data_root}. "
            "Check participants.tsv and subject/anat file structure."
        )

    return samples


def make_client_fn(
    client_train_splits: list[list[dict]],
    client_val_splits: list[list[dict]],
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    local_epochs: int = 1,
    image_size: tuple[int, int, int] = (96, 96, 96),
):
    def client_fn(cid: str):
        client_id = int(cid)
        return MRIClassificationClient(
            train_samples=client_train_splits[client_id],
            val_samples=client_val_splits[client_id],
            batch_size=batch_size,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            image_size=image_size,
        ).to_client()

    return client_fn


def _history_to_dict(history) -> dict:
    return {
        "losses_distributed": history.losses_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_distributed_fit": history.metrics_distributed_fit,
        "metrics_distributed": history.metrics_distributed,
        "metrics_centralized": history.metrics_centralized,
    }


def _history_to_dataframe(history) -> pd.DataFrame:
    round_map: dict[int, dict] = {}

    for rnd, loss in history.losses_distributed:
        round_map.setdefault(rnd, {"round": rnd})
        round_map[rnd]["loss_distributed"] = loss

    for metric_name, values in history.metrics_distributed_fit.items():
        for rnd, val in values:
            round_map.setdefault(rnd, {"round": rnd})
            round_map[rnd][metric_name] = val

    for metric_name, values in history.metrics_distributed.items():
        for rnd, val in values:
            round_map.setdefault(rnd, {"round": rnd})
            round_map[rnd][metric_name] = val

    rows = [round_map[r] for r in sorted(round_map.keys())]
    return pd.DataFrame(rows)


def save_fl_results(
    history,
    out_dir: str,
    experiment_name: str,
    config_dict: dict,
    client_summary: dict,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{experiment_name}_history.json")
    csv_path = os.path.join(out_dir, f"{experiment_name}_metrics.csv")

    payload = {
        "experiment_name": experiment_name,
        "config": config_dict,
        "client_summary": client_summary,
        "history": _history_to_dict(history),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    df = _history_to_dataframe(history)
    df.to_csv(csv_path, index=False)

    print(f"Saved federated history to: {json_path}")
    print(f"Saved federated metrics to: {csv_path}")
    print("JSON exists:", os.path.exists(json_path))
    print("CSV exists:", os.path.exists(csv_path))

    return json_path, csv_path


def main():
    set_seed(42)

    data_root = "/content/drive/MyDrive/Internship_MONAI/data_zips/ds005892-download"
    out_dir = "/content/drive/MyDrive/Internship_MONAI/fl_results"
    experiment_name = "parkinson_fl_2clients_3rounds"

    num_clients = 2
    num_rounds = 3
    batch_size = 2
    learning_rate = 1e-4
    local_epochs = 1
    image_size = (96, 96, 96)

    samples = load_parkinson_samples(data_root=data_root)
    print(f"Total samples: {len(samples)}")

    train_samples, val_samples = split_classification_data(
        samples=samples,
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    print(f"Global train samples: {len(train_samples)}")
    print(f"Global val samples: {len(val_samples)}")

    client_train_splits = split_samples_iid(train_samples, num_clients=num_clients, seed=42)
    client_val_splits = split_samples_iid(val_samples, num_clients=num_clients, seed=42)

    print("\nClient train summary:")
    print_client_summary(client_train_splits)

    print("\nClient val summary:")
    print_client_summary(client_val_splits)

    client_summary = {
        "train": [
            {
                "client_id": i,
                "num_samples": len(split),
                "class_distribution": {
                    str(k): sum(1 for x in split if x["label"] == k)
                    for k in sorted(set(x["label"] for x in split))
                },
            }
            for i, split in enumerate(client_train_splits)
        ],
        "val": [
            {
                "client_id": i,
                "num_samples": len(split),
                "class_distribution": {
                    str(k): sum(1 for x in split if x["label"] == k)
                    for k in sorted(set(x["label"] for x in split))
                },
            }
            for i, split in enumerate(client_val_splits)
        ],
    }

    strategy = build_fedavg_strategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    client_fn = make_client_fn(
        client_train_splits=client_train_splits,
        client_val_splits=client_val_splits,
        batch_size=batch_size,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        image_size=image_size,
    )

    if ray.is_initialized():
        ray.shutdown()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    config_dict = {
        "data_root": data_root,
        "out_dir": out_dir,
        "experiment_name": experiment_name,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "local_epochs": local_epochs,
        "image_size": image_size,
    }

    json_path, csv_path = save_fl_results(
        history=history,
        out_dir=out_dir,
        experiment_name=experiment_name,
        config_dict=config_dict,
        client_summary=client_summary,
    )

    return {
        "json_path": json_path,
        "csv_path": csv_path,
    }


if __name__ == "__main__":
    main()
