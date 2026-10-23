"""
Unit tests for CVAE architecture, tensor contracts, backprop, and inference routines.
"""
import pytest
import torch
import torch.nn as nn
from utils.model import (
    CVAE, cvae_loss, load_model, predict_image,
    generate_image, generate_batch,
    IMG_SIZE, LATENT_DIM, HIDDEN_DIM, NUM_CLASSES, INPUT_DIM
)


def test_model_initialization(model):
    """Verify layer configurations and parameter allocations."""
    assert isinstance(model.fc1, nn.Linear)
    assert model.fc1.in_features == INPUT_DIM + NUM_CLASSES
    assert model.fc1.out_features == HIDDEN_DIM

    assert model.fc_mu.out_features == LATENT_DIM
    assert model.fc_logvar.out_features == LATENT_DIM

    assert model.fc2.in_features == LATENT_DIM + NUM_CLASSES
    assert model.fc2.out_features == HIDDEN_DIM

    assert model.fc3.out_features == INPUT_DIM


def test_encoder_contract(model, device):
    """Ensure encoder maps (B, 12288) and (B, 2) to correct Gaussian parameters."""
    batch_size = 4
    x = torch.rand(batch_size, INPUT_DIM, device=device)
    y = torch.zeros(batch_size, NUM_CLASSES, device=device)
    y[:, 0] = 1.0  # Healthy class

    mu, logvar = model.encode(x, y)
    assert mu.shape == (batch_size, LATENT_DIM)
    assert logvar.shape == (batch_size, LATENT_DIM)


def test_reparameterization_trick(model, device):
    """Ensure stochastic latent sampling retains shape and is differentiable."""
    mu = torch.zeros(3, LATENT_DIM, device=device, requires_grad=True)
    logvar = torch.zeros(3, LATENT_DIM, device=device, requires_grad=True)

    z = model.reparameterize(mu, logvar)
    assert z.shape == (3, LATENT_DIM)

    # Check backprop through reparameterization
    loss = z.sum()
    loss.backward()
    assert mu.grad is not None
    assert logvar.grad is not None


def test_decoder_activation_bounds(model, device):
    """Check that decoder output is strictly within [0, 1] through Sigmoid."""
    z = torch.randn(2, LATENT_DIM, device=device)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)

    recon = model.decode(z, y)
    assert recon.shape == (2, INPUT_DIM)
    assert torch.all(recon >= 0.0)
    assert torch.all(recon <= 1.0)


def test_full_forward_and_backward_pass(device):
    """Verify complete end-to-end forward/backward cycle and gradient updates."""
    m = CVAE().to(device)
    m.train()
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

    x = torch.rand(4, INPUT_DIM, device=device)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], device=device)

    recon, mu, logvar = m(x, y)
    loss, bce, kld = cvae_loss(recon, x, mu, logvar)

    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert kld.item() >= 0.0

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Confirm all layers received gradients
    for name, p in m.named_parameters():
        assert p.grad is not None, f"Parameter {name} did not receive gradients"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradients"


def test_predict_image_shapes_and_metrics(model, device):
    """Test inference helper with single and batched tensors."""
    threshold = 0.014188

    # 4D tensor (1, 3, 64, 64)
    img_4d = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    res_4d = predict_image(model, img_4d, class_idx=0, threshold=threshold, device=device)

    assert "error" in res_4d
    assert "prediction" in res_4d
    assert res_4d["prediction"] in ["Healthy", "Stressed"]
    assert 0.0 <= res_4d["confidence"] <= 99.9
    assert res_4d["recon_img"].shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert res_4d["residual_map"].shape == (1, 1, IMG_SIZE, IMG_SIZE)

    # 3D tensor (3, 64, 64)
    img_3d = torch.rand(3, IMG_SIZE, IMG_SIZE)
    res_3d = predict_image(model, img_3d, class_idx=1, threshold=threshold, device=device)
    assert res_3d["recon_img"].shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_generation_helpers(model, device):
    """Test single sample and batched generative decoding."""
    single_img = generate_image(model, class_idx=0, device=device, seed=42)
    assert single_img.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert torch.all(single_img >= 0.0) and torch.all(single_img <= 1.0)

    batch_imgs = generate_batch(model, class_idx=1, num_samples=3, device=device)
    assert batch_imgs.shape == (3, 3, IMG_SIZE, IMG_SIZE)
    assert torch.all(batch_imgs >= 0.0) and torch.all(batch_imgs <= 1.0)


def test_load_model_resilience():
    """Verify fallback initialization when checkpoint file does not exist."""
    m, d = load_model("nonexistent_weights.pth", fallback_to_initialized=True)
    assert isinstance(m, CVAE)
    assert d is not None

    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_weights.pth", fallback_to_initialized=False)
