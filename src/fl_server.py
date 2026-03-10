from __future__ import annotations

from typing import Callable, Optional

import flwr as fl


def weighted_average(metrics):
    """
    Aggregate client metrics weighted by the number of examples.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}
    metric_names = set()
    for _, client_metrics in metrics:
        metric_names.update(client_metrics.keys())

    for name in metric_names:
        weighted_sum = 0.0
        for num_examples, client_metrics in metrics:
            value = client_metrics.get(name)
            if value is not None:
                weighted_sum += num_examples * float(value)
        aggregated[name] = weighted_sum / total_examples

    return aggregated


def build_fedavg_strategy(
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
):
    """
    Build a basic FedAvg strategy for classification experiments.
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    return strategy


def start_server(
    num_rounds: int = 3,
    server_address: str = "0.0.0.0:8080",
):
    """
    Start a simple Flower server with FedAvg.
    """
    strategy = build_fedavg_strategy()

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
