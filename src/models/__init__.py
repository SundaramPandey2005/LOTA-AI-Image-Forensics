from .backbones import get_backbone
from .nbc import NoiseBasedClassifier, LOTANoiseClassifier
from .ngc import NoiseGuidedClassifier


def create_model(config: dict):
    """
    Model factory matching architecture string from experiment config.
    """
    arch = config.get("model", {}).get("architecture", "nbc").lower()
    backbone_name = config.get("model", {}).get("backbone", "resnet50")
    pretrained = config.get("model", {}).get("pretrained", True)
    num_classes = config.get("model", {}).get("num_classes", 1)

    if arch == "nbc":
        return NoiseBasedClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    elif arch == "ngc":
        return NoiseGuidedClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model architecture: '{arch}'. Choose 'nbc' or 'ngc'.")


__all__ = [
    "get_backbone",
    "NoiseBasedClassifier",
    "LOTANoiseClassifier",
    "NoiseGuidedClassifier",
    "create_model",
]
