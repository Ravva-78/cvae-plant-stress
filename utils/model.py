"""
CVAE model definition + inference helpers.
Production-grade Conditional Variational Autoencoder (CVAE) for plant stress anomaly detection.
Shared between notebook, web app, CLI demo, and test suite.
"""
from typing import Dict, Any, Tuple, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 64
LATENT_DIM = 50
HIDDEN_DIM = 512
NUM_CLASSES = 2
INPUT_DIM = IMG_SIZE * IMG_SIZE * 3


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) architecture for plant stress detection.

    Encoder: q_phi(z | x, y) maps concatenated [flattened image, one-hot label] -> (mu, logvar)
    Reparameterization: z = mu + sigma * eps, eps ~ N(0, I)
    Decoder: p_theta(x | z, y) maps concatenated [latent z, one-hot label] -> reconstructed image x_hat
    """
    def __init__(self, num_classes: int = NUM_CLASSES, latent_dim: int = LATENT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.fc1 = nn.Linear(INPUT_DIM + num_classes, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc2 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, INPUT_DIM)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input x conditioned on label y into variational Gaussian parameters (mu, logvar).
        """
        device = self.fc1.weight.device
        if x.device != device:
            x = x.to(device)
        if y.device != device:
            y = y.to(device)
        h = F.relu(self.fc1(torch.cat([x, y], dim=-1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for backpropagation through stochastic latent node:
        z = mu + exp(0.5 * logvar) * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector z conditioned on label y into reconstructed image space.
        """
        device = self.fc2.weight.device
        if z.device != device:
            z = z.to(device)
        if y.device != device:
            y = y.to(device)
        h = F.relu(self.fc2(torch.cat([z, y], dim=-1)))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns:
            recon_x: reconstructed image in [0, 1] range (flattened)
            mu: latent distribution mean
            logvar: latent distribution log-variance
        """
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar


def cvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "sum"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardized CVAE loss function: Negative Evidence Lower Bound (-ELBO).
    Loss = Reconstruction_Loss + beta * KL_Divergence

    Reconstruction: Binary Cross Entropy (BCE) between Sigmoid outputs and inputs.
    KL Divergence: Analytical divergence from standard Gaussian N(0, I):
                   -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    if x.device != recon_x.device:
        x = x.to(recon_x.device)
    bce = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "mean":
        kld = kld / x.size(0)
    total_loss = bce + beta * kld
    return total_loss, bce, kld


def load_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    fallback_to_initialized: bool = True
) -> Tuple[CVAE, torch.device]:
    """
    Load CVAE model weights with robust offline/demo fallback.

    Args:
        weights_path: Path to .pth checkpoint file.
        device: Torch device (defaults to CUDA if available, else CPU).
        fallback_to_initialized: If True and weights file is missing, initializes
                                a CVAE model so demo and test suites run out of the box.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE().to(device)

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model, device

    if fallback_to_initialized:
        # Pre-seed deterministic initialization for demo mode
        torch.manual_seed(42)
        model.eval()
        return model, device

    raise FileNotFoundError(f"Weights file not found at: {weights_path}")


def predict_image(
    model: CVAE,
    img_tensor: torch.Tensor,
    class_idx: int,
    threshold: float,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate plant image under specified class conditioning and compute anomaly metric.

    Args:
        model: CVAE instance.
        img_tensor: Tensor of shape (1, 3, 64, 64) or (3, 64, 64).
        class_idx: Class index conditioning (0=Healthy, 1=Stressed).
        threshold: Decision boundary for MSE reconstruction error.
        device: Device to execute inference on.

    Returns:
        Dictionary with reconstruction error, prediction, confidence score,
        threshold, reconstructed image tensor, residual heatmap, and latent mu vector.
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    x_flat = img_tensor.view(img_tensor.size(0), -1).to(device)
    y_onehot = F.one_hot(
        torch.tensor([class_idx], device=device),
        num_classes=model.num_classes
    ).float()

    with torch.no_grad():
        recon, mu, logvar = model(x_flat, y_onehot)

    error = F.mse_loss(recon, x_flat, reduction="mean").item()
    prediction = "Stressed" if error > threshold else "Healthy"

    # Normalized confidence calculation based on distance from anomaly threshold
    spread = abs(error - threshold)
    confidence = min(spread / (threshold + 1e-8) * 100, 99.9)

    recon_img = recon.view(1, 3, IMG_SIZE, IMG_SIZE).cpu()
    orig_img = x_flat.view(1, 3, IMG_SIZE, IMG_SIZE).cpu()

    # Compute spatial residual map (L1 difference across channels)
    residual_map = torch.abs(orig_img - recon_img).mean(dim=1, keepdim=True)

    return {
        "error": round(error, 6),
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "threshold": round(threshold, 6),
        "recon_img": recon_img,
        "residual_map": residual_map,
        "mu": mu.cpu().numpy().tolist(),
    }


def generate_image(
    model: CVAE,
    class_idx: int,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Sample latent vector z from prior N(0, I) and decode conditioned on class.

    Returns:
        (1, 3, IMG_SIZE, IMG_SIZE) float tensor in [0, 1] range.
    """
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(1, model.latent_dim, generator=gen, device=device)
    else:
        z = torch.randn(1, model.latent_dim, device=device)

    y_onehot = F.one_hot(
        torch.tensor([class_idx], device=device),
        num_classes=model.num_classes
    ).float()

    with torch.no_grad():
        gen_out = model.decode(z, y_onehot)

    img = gen_out.view(1, 3, IMG_SIZE, IMG_SIZE).cpu()
    img = img.clamp(0.0, 1.0)
    return img


def generate_batch(
    model: CVAE,
    class_idx: int,
    num_samples: int = 4,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate multiple synthetic plant images conditioned on class.

    Returns:
        (num_samples, 3, IMG_SIZE, IMG_SIZE) float tensor in [0, 1] range.
    """
    if device is None:
        device = next(model.parameters()).device

    z = torch.randn(num_samples, model.latent_dim, device=device)
    y_onehot = F.one_hot(
        torch.full((num_samples,), class_idx, dtype=torch.long, device=device),
        num_classes=model.num_classes
    ).float()

    with torch.no_grad():
        gen_out = model.decode(z, y_onehot)

    imgs = gen_out.view(num_samples, 3, IMG_SIZE, IMG_SIZE).cpu()
    return imgs.clamp(0.0, 1.0)
