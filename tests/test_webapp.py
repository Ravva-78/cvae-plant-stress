"""
Integration tests for web application routes, API responses, and inference serving.
"""
import json


def test_health_endpoint(web_client):
    """Ensure /health returns 200 with service status."""
    res = web_client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert data["model_initialized"] is True


def test_index_route(web_client):
    """Ensure landing page serves cleanly with UI HTML."""
    res = web_client.get("/")
    assert res.status_code == 200
    assert b"Plant Stress Detector" in res.data


def test_metrics_route(web_client):
    """Ensure /metrics serves experimental benchmark values."""
    res = web_client.get("/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "accuracy" in data
    assert "auc" in data
    assert "threshold" in data


def test_serve_results_static(web_client):
    """Ensure generated result assets are served statically."""
    res = web_client.get("/results/metrics.json")
    assert res.status_code == 200

    res_img = web_client.get("/results/confusion_matrix.png")
    assert res_img.status_code == 200


def test_predict_endpoint_missing_payload(web_client):
    """POST /predict without file must reject with 400 Bad Request."""
    res = web_client.post("/predict")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_endpoint_success(web_client, sample_png_bytes):
    """POST /predict with valid image stream executes end-to-end inference."""
    data = {
        "image": (sample_png_bytes, "sample_leaf.png")
    }
    res = web_client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    body = res.get_json()

    assert "prediction" in body
    assert body["prediction"] in ["Healthy", "Stressed"]
    assert "confidence" in body
    assert "error" in body
    assert "threshold" in body
    assert "original_b64" in body
    assert "recon_b64" in body
    assert "residual_b64" in body


def test_generate_endpoint_success(web_client):
    """POST /generate generates conditioned synthetic plant phenotype."""
    # Test Healthy class
    res = web_client.post(
        "/generate",
        data=json.dumps({"class": "Healthy"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["class"] == "Healthy"
    assert "image_b64" in body
    assert len(body["image_b64"]) > 50

    # Test Stressed class
    res_s = web_client.post(
        "/generate",
        data=json.dumps({"class": "Stressed"}),
        content_type="application/json"
    )
    assert res_s.status_code == 200
    assert res_s.get_json()["class"] == "Stressed"
