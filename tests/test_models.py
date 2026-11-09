import copy

import numpy as np
import pytest
import torch

from src.evaluation.evaluator import metrics_at_k
from src.models.multi_patch import MultiPatchLOTA, aggregate_logits
from src.training.engine import load_checkpoint, train_model
from src.training.losses import multi_patch_loss
from src.utils.config import load_config


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return MultiPatchLOTA("resnet18", pretrained=False, num_patches=4, input_size=64).eval()


def patches(batch=2, k=4):
    return torch.randint(0, 256, (batch, k, 3, 32, 32), dtype=torch.uint8)


def test_output_shapes(model):
    image_logit, patch_logits = model(patches())
    assert image_logit.shape == (2,) and patch_logits.shape == (2, 4)


def test_k1_is_exactly_the_original_patch_classifier(model):
    """LOTA-MP with K = 1 must reduce to the baseline decision."""
    single = copy.deepcopy(model)
    single.num_patches = 1
    x = patches()
    with torch.no_grad():
        image_logit, _ = single(x)
        reference = single.classifier(x[:, 0])
    assert torch.allclose(image_logit, reference, atol=1e-6)


def test_fused_logit_is_the_mean_of_shared_backbone_logits(model):
    x = patches()
    with torch.no_grad():
        image_logit, patch_logits = model(x)
        separately = torch.stack([model.classifier(x[:, i]) for i in range(4)], dim=1)
    assert torch.allclose(patch_logits, separately, atol=1e-5)
    assert torch.allclose(image_logit, separately.mean(dim=1), atol=1e-5)


def test_no_parameters_are_added(model):
    baseline = MultiPatchLOTA("resnet18", pretrained=False, num_patches=1, input_size=64)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(model) == count(baseline)
    assert model.state_dict().keys() == baseline.state_dict().keys()


def test_aggregation_rules():
    logits = torch.tensor([[2.0, -1.0, 3.0, 0.5]])
    assert torch.allclose(aggregate_logits(logits, "logit_mean"), torch.tensor([1.125]))
    prob = torch.sigmoid(aggregate_logits(logits, "prob_mean"))
    assert torch.allclose(prob, torch.sigmoid(logits).mean(dim=1), atol=1e-5)
    assert torch.allclose(torch.sigmoid(aggregate_logits(logits, "vote")), torch.tensor([0.75]), atol=1e-5)
    with pytest.raises(ValueError):
        aggregate_logits(logits, "median")


def test_variance_of_the_mean_logit_follows_the_derivation():
    """Var[mean of K logits] = sigma^2 (rho + (1 - rho) / K) for equicorrelated logits."""
    rng = np.random.default_rng(0)
    sigma, rho, k, n = 2.0, 0.3, 4, 200_000
    shared = rng.normal(size=(n, 1)) * np.sqrt(rho)
    private = rng.normal(size=(n, k)) * np.sqrt(1 - rho)
    logits = sigma * (shared + private)
    expected = sigma**2 * (rho + (1 - rho) / k)
    assert abs(logits.mean(axis=1).var() - expected) < 0.03


def test_fusion_improves_accuracy_on_simulated_noisy_patches():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 4000)
    logits = (2 * labels[:, None] - 1) * 1.0 + rng.normal(size=(4000, 8)) * 2.0
    acc = [metrics_at_k(logits.astype(np.float32), labels, k)["acc"] for k in (1, 4, 8)]
    assert acc[0] < acc[1] < acc[2]


def test_loss_reduces_to_bce_for_k1_and_obeys_jensen():
    labels = torch.tensor([1.0, 0.0, 1.0])
    logits = torch.randn(3, 4)
    bag = multi_patch_loss(logits.mean(dim=1), logits, labels, 0.0)
    both = multi_patch_loss(logits.mean(dim=1), logits, labels, 1.0)
    instance = both - bag
    assert instance >= bag - 1e-6  # BCE is convex in the logit
    single = logits[:, :1]
    assert torch.allclose(multi_patch_loss(single[:, 0], single, labels, 1.0),
                          2 * multi_patch_loss(single[:, 0], single, labels, 0.0))


def test_train_and_reload_roundtrip(tiny_genimage, tmp_path):
    config = load_config(None, [
        "model.backbone=resnet18", "model.pretrained=false", "model.num_patches=2",
        "forensic.image_size=64", "train.epochs=1", "train.batch_size=8", "eval.batch_size=8",
        "data.num_workers=0", "data.val_fraction=0.25",
    ])
    config["data"]["root"] = tiny_genimage
    summary = train_model(config, str(tmp_path), torch.device("cpu"))
    assert summary["best_epoch"] == 1 and len(summary["history"]) == 1
    reloaded, checkpoint = load_checkpoint(str(tmp_path / "best.pth"), torch.device("cpu"))
    assert reloaded.num_patches == 2 and checkpoint["epoch"] == 1
