from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd
import flwr as fl

from src.data import split_classification_data
from src.fl_client import MRIClassificationClient
from src.fl_server import build_fedavg_strategy
from src.fl_utils import print_client_summary, split_samples_iid
from src.utils import set_seed


def load_parkinson_samples(data_root: str) -> list[dict]:
    """
    Expected OpenNeuro/BIDS-style structure:
    data_root/
        participants.tsv
        sub-XXXX/
            anat/
                sub-XXXX_T1w.nii.gz
    """

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


def main():
    set_seed(42)

    # CHANGE THIS PATH TO YOUR LOCAL DATASET LOCATION
    data_root = "/content/drive/MyDrive/Internship_MONAI/data_zips/ds005892-download"

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

    client_train_splits = split_samples_iid(
        train_samples,
        num_clients=num_clients,
        seed=42,
    )
    client_val_splits = split_samples_iid(
        val_samples,
        num_clients=num_clients,
        seed=42,
    )

    print("\nClient train summary:")
    print_client_summary(client_train_splits)

    print("\nClient val summary:")
    print_client_summary(client_val_splits)

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

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
