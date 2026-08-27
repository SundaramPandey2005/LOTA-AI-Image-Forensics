import pytest
import torch

from src.models.nbc import NoiseBasedClassifier
from src.models.ngc import NoiseGuidedClassifier
from src.models import create_model


class TestModels:
    def test_nbc_forward_and_backward(self):
        # NBC without pretrained weights for fast testing
        model = NoiseBasedClassifier(backbone_name="resnet18", pretrained=False, num_classes=1)
        model.train()

        B = 2
        noise_patch = torch.rand((B, 3, 32, 32), dtype=torch.float32) * 255.0
        labels = torch.tensor([0.0, 1.0], dtype=torch.float32)

        logits = model(noise_patch)
        assert logits.shape == (B,)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        # Check gradients exist on classifier head
        assert model.fc.weight.grad is not None
        assert model.fc.weight.grad.norm().item() > 0

    def test_ngc_forward_and_backward(self):
        # NGC without pretrained weights for fast testing
        model = NoiseGuidedClassifier(backbone_name="resnet18", pretrained=False, num_classes=1, num_heads=4)
        model.train()

        B = 2
        noise_patch = torch.rand((B, 3, 32, 32), dtype=torch.float32) * 255.0
        raw_image = torch.rand((B, 3, 256, 256), dtype=torch.float32) * 255.0
        labels = torch.tensor([0.0, 1.0], dtype=torch.float32)

        logits = model(noise_patch, raw_image)
        assert logits.shape == (B,)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        # Check gradients exist on cross attention and classifier
        assert model.classifier.weight.grad is not None
        assert model.classifier.weight.grad.norm().item() > 0

    def test_create_model_factory(self):
        cfg_nbc = {"model": {"architecture": "nbc", "backbone": "resnet18", "pretrained": False}}
        m_nbc = create_model(cfg_nbc)
        assert isinstance(m_nbc, NoiseBasedClassifier)

        cfg_ngc = {"model": {"architecture": "ngc", "backbone": "resnet18", "pretrained": False}}
        m_ngc = create_model(cfg_ngc)
        assert isinstance(m_ngc, NoiseGuidedClassifier)
