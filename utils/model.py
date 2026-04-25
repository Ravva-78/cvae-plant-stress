"""
CVAE model definition + inference helpers.
Shared between notebook and web app.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE   = 64
LATENT_DIM = 50
HIDDEN_DIM = 512
NUM_CLASSES = 2
INPUT_DIM  = IMG_SIZE * IMG_SIZE * 3


class CVAE(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.fc1       = nn.Linear(INPUT_DIM + num_classes, HIDDEN_DIM)
        self.fc_mu     = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc2       = nn.Linear(LATENT_DIM + num_classes, HIDDEN_DIM)
        self.fc3       = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def encode(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y):
        h = F.relu(self.fc2(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


def load_model(weights_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def predict_image(model, img_tensor, class_idx: int, threshold: float, device):
    """
    img_tensor : (1, 3, 64, 64) float tensor
    class_idx  : 0=Healthy, 1=Stressed (ImageFolder order)
    Returns    : dict with error, prediction, confidence
    """
    x_flat   = img_tensor.view(1, -1).to(device)
    y_onehot = F.one_hot(
        torch.tensor([class_idx]), num_classes=NUM_CLASSES
    ).float().to(device)

    with torch.no_grad():
        recon, mu, logvar = model(x_flat, y_onehot)

    error = F.mse_loss(recon, x_flat, reduction="mean").item()
    prediction = "Stressed" if error > threshold else "Healthy"

    # Simple confidence: distance from threshold normalised
    spread = abs(error - threshold)
    confidence = min(spread / (threshold + 1e-8) * 100, 99.9)

    recon_img = recon.view(1, 3, IMG_SIZE, IMG_SIZE).cpu()

    return {
        "error"      : round(error, 6),
        "prediction" : prediction,
        "confidence" : round(confidence, 1),
        "threshold"  : round(threshold, 6),
        "recon_img"  : recon_img,
        "mu"         : mu.cpu().numpy().tolist(),
    }


def generate_image(model, class_idx: int, device):
    """Sample from prior and decode to generate a synthetic plant image."""
    z        = torch.randn(1, LATENT_DIM).to(device)
    y_onehot = F.one_hot(
        torch.tensor([class_idx]), num_classes=NUM_CLASSES
    ).float().to(device)
    with torch.no_grad():
        gen = model.decode(z, y_onehot)
    img = gen.view(1, 3, IMG_SIZE, IMG_SIZE).cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return img
