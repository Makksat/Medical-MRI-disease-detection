from __future__ import annotations

import os
import random
from typing import Sequence

from sklearn.model_selection import train_test_split


def list_subfolders(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))
        ]
    )


def build_classification_samples_from_class_folders(
    root_dir: str,
    class_to_label: dict[str, int],
    extensions: Sequence[str] = (".nii", ".nii.gz"),
) -> list[dict]:
    """
    Expected structure:
    root_dir/
        class_a/
            sample1.nii.gz
            sample2.nii.gz
        class_b/
            sample3.nii.gz
    """
    samples = []

    for class_name, label in class_to_label.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: class folder not found: {class_dir}")
            continue

        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(tuple(extensions)):
                samples.append(
                    {
                        "image": os.path.join(class_dir, fname),
                        "label": label,
                        "class_name": class_name,
                        "filename": fname,
                    }
                )

    return samples


def split_classification_data(
    samples: list[dict],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[list[dict], list[dict]]:
    if len(samples) == 0:
        raise ValueError("No samples provided for train/val split.")

    labels = [item["label"] for item in samples]
    stratify_labels = labels if stratify else None

    train_samples, val_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return train_samples, val_samples


def split_into_clients(
    samples: list[dict],
    num_clients: int = 2,
    shuffle: bool = True,
    seed: int = 42,
) -> list[list[dict]]:
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    items = samples.copy()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    client_splits = [[] for _ in range(num_clients)]
    for idx, item in enumerate(items):
        client_splits[idx % num_clients].append(item)

    return client_splits


def summarize_class_distribution(samples: list[dict]) -> dict:
    summary = {}
    for item in samples:
        label = item["label"]
        summary[label] = summary.get(label, 0) + 1
    return summary


def build_msseg_samples(
    train_path: str,
    required_modalities: Sequence[str] = ("flair", "t1", "t2", "dp"),
    image_folder_name: str = "Preprocessed_Data",
    mask_folder_name: str = "Masks",
    mask_keyword: str = "consensus",
) -> list[dict]:
    """
    Expected structure:
    train_path/
        Center_XX/
            Patient_XX/
                Preprocessed_Data/
                Masks/
    """
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"MSSEG train path not found: {train_path}")

    centers = sorted(
        [
            name
            for name in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, name))
        ]
    )

    data_dicts = []

    for center in centers:
        center_path = os.path.join(train_path, center)
        patients = sorted(
            [
                name
                for name in os.listdir(center_path)
                if os.path.isdir(os.path.join(center_path, name))
            ]
        )

        for patient in patients:
            patient_path = os.path.join(center_path, patient)
            img_dir = os.path.join(patient_path, image_folder_name)
            mask_dir = os.path.join(patient_path, mask_folder_name)

            if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
                continue

            img_files = sorted(os.listdir(img_dir))
            mask_files = sorted(os.listdir(mask_dir))

            def find_modality(modality_name: str) -> str | None:
                matches = [
                    os.path.join(img_dir, fname)
                    for fname in img_files
                    if modality_name in fname.lower() and fname.endswith(".nii.gz")
                ]
                return matches[0] if matches else None

            image_paths = [find_modality(mod) for mod in required_modalities]

            mask_candidates = [
                os.path.join(mask_dir, fname)
                for fname in mask_files
                if mask_keyword in fname.lower() and fname.endswith(".nii.gz")
            ]
            label_path = mask_candidates[0] if mask_candidates else None

            if all(path is not None for path in image_paths) and label_path is not None:
                data_dicts.append(
                    {
                        "image": image_paths,
                        "label": label_path,
                        "center": center,
                        "patient": patient,
                    }
                )

    return data_dicts
