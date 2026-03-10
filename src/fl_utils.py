from __future__ import annotations

import random
from collections import Counter


def split_samples_iid(
    samples: list[dict],
    num_clients: int = 2,
    seed: int = 42,
) -> list[list[dict]]:
    """
    Split samples approximately evenly across clients (IID-style).
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    items = samples.copy()
    rng = random.Random(seed)
    rng.shuffle(items)

    client_splits = [[] for _ in range(num_clients)]
    for idx, item in enumerate(items):
        client_splits[idx % num_clients].append(item)

    return client_splits


def split_samples_non_iid_by_class(
    samples: list[dict],
    num_clients: int = 2,
    seed: int = 42,
) -> list[list[dict]]:
    """
    Simple non-IID split:
    group by class, shuffle within each class, then assign class-heavy partitions.
    Works best for binary classification experiments.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    rng = random.Random(seed)

    class_groups = {}
    for item in samples:
        label = item["label"]
        class_groups.setdefault(label, []).append(item)

    for label in class_groups:
        rng.shuffle(class_groups[label])

    clients = [[] for _ in range(num_clients)]
    labels = sorted(class_groups.keys())

    if len(labels) == 0:
        return clients

    if len(labels) == 1:
        only_items = class_groups[labels[0]]
        for idx, item in enumerate(only_items):
            clients[idx % num_clients].append(item)
        return clients

    # For binary case, bias earlier clients toward class 0 and later clients toward class 1
    if len(labels) == 2:
        label0, label1 = labels[0], labels[1]
        items0 = class_groups[label0]
        items1 = class_groups[label1]

        split0 = len(items0) // 2
        split1 = len(items1) // 2

        first_half = num_clients // 2
        second_half = num_clients - first_half

        for idx, item in enumerate(items0[:split0]):
            clients[idx % max(first_half, 1)].append(item)

        for idx, item in enumerate(items0[split0:]):
            clients[first_half + (idx % max(second_half, 1))].append(item)

        for idx, item in enumerate(items1[:split1]):
            clients[first_half + (idx % max(second_half, 1))].append(item)

        for idx, item in enumerate(items1[split1:]):
            clients[idx % max(first_half, 1)].append(item)

        return clients

    # Fallback for multi-class
    all_items = samples.copy()
    rng.shuffle(all_items)
    for idx, item in enumerate(all_items):
        clients[idx % num_clients].append(item)

    return clients


def summarize_client_partitions(client_splits: list[list[dict]]) -> list[dict]:
    """
    Return summary stats per client.
    """
    summaries = []
    for client_id, split in enumerate(client_splits):
        labels = [item["label"] for item in split]
        summaries.append(
            {
                "client_id": client_id,
                "num_samples": len(split),
                "class_distribution": dict(Counter(labels)),
            }
        )
    return summaries


def print_client_summary(client_splits: list[list[dict]]) -> None:
    summaries = summarize_client_partitions(client_splits)
    for summary in summaries:
        print(
            f"Client {summary['client_id']}: "
            f"n={summary['num_samples']}, "
            f"class_distribution={summary['class_distribution']}"
        )
