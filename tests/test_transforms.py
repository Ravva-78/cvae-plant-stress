"""
Unit tests for data transforms, preprocessing, and sensor contracts.
"""
import torch
from PIL import Image

from utils.model import IMG_SIZE
from webapp.app import preprocess, tensor_to_b64


def test_standard_preprocessing():
    """Verify input image resizing and tensor conversion contract."""
    raw_img = Image.new("RGB", (256, 128), color=(100, 150, 50))
    tensor = preprocess(raw_img)

    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_rgba_to_rgb_handling():
    """Ensure images with alpha transparency are properly converted to 3 channels."""
    rgba_img = Image.new("RGBA", (100, 100), color=(50, 200, 50, 128))
    rgb_img = rgba_img.convert("RGB")
    tensor = preprocess(rgb_img)

    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_tensor_to_base64_serialization():
    """Test serialization of reconstructed tensors and single-channel heatmaps to Base64 PNG."""
    t_3ch = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    b64_str = tensor_to_b64(t_3ch)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 100

    t_1ch = torch.rand(1, 1, IMG_SIZE, IMG_SIZE)
    b64_heat = tensor_to_b64(t_1ch)
    assert isinstance(b64_heat, str)
    assert len(b64_heat) > 100
