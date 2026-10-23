"""
Shared pytest fixtures for CVAE test suite.
"""
import io
import pytest
import torch
import numpy as np
from PIL import Image

from utils.model import CVAE, IMG_SIZE
from webapp.app import app


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    torch.manual_seed(42)
    m = CVAE().to(device)
    m.eval()
    return m


@pytest.fixture
def sample_leaf_tensor():
    """Synthetic RGB leaf tensor (1, 3, 64, 64) with values in [0, 1]."""
    torch.manual_seed(101)
    return torch.rand(1, 3, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def sample_png_bytes():
    """In-memory PNG image file bytes."""
    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def web_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
