"""
Unit tests validating configuration schemas, metrics ranges, and experimental integrity.
"""
import os
import json


def test_metrics_json_schema():
    """Validate structure, types, and boundaries of results/metrics.json."""
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.json")
    assert os.path.exists(metrics_path), "metrics.json must exist in results/"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Required keys
    required_keys = [
        "accuracy", "precision", "recall", "f1_score", "auc",
        "threshold", "classes", "n_train", "n_test", "epochs",
        "latent_dim", "train_losses"
    ]
    for k in required_keys:
        assert k in metrics, f"Missing required key '{k}' in metrics.json"

    # Value boundaries
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["threshold"] > 0.0

    # Dataset integrity
    assert metrics["classes"] == ["Healthy", "Stressed"]
    assert metrics["n_train"] == 4901
    assert metrics["n_test"] == 1226
    assert metrics["epochs"] == len(metrics["train_losses"])
    assert metrics["latent_dim"] == 50

    # Loss convergence trend: first loss > last loss
    assert metrics["train_losses"][0] > metrics["train_losses"][-1]
