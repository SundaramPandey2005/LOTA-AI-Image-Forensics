from typing import Tuple, List, Union, Optional, Callable
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from src.forensic.bitplanes import compose_low_bit_planes
from src.forensic.normalization import normalize_noise_thresholding, normalize_noise_scaling
from src.forensic.mgps import maximum_gradient_patch_selection

# Standard ImageNet normalization parameters for classification backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_for_backbone(
    image: torch.Tensor,
    input_is_255: bool = True
) -> torch.Tensor:
    """
    Apply standard ImageNet mean and std normalization to a tensor.

    Args:
        image (torch.Tensor): Tensor of shape (..., 3, H, W).
        input_is_255 (bool): If True, divides input by 255.0 first.

    Returns:
        torch.Tensor: Normalized tensor with zero mean and unit variance.
    """
    x = image / 255.0 if input_is_255 else image
    
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    
    return (x - mean) / std


def get_transforms(
    image_size: int = 256,
    is_training: bool = True
) -> Callable:
    """
    Standard preprocessing transforms.
    Resizes image to image_size (default 256x256) via bilinear interpolation.
    """
    transform_list = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
    ]
    return T.Compose(transform_list)


def preprocess_raw_image(
    image: Image.Image,
    image_size: int = 256,
    pre_resize_size: Optional[int] = None
) -> torch.Tensor:
    """
    Convert a PIL image to a (3, H, W) float32 PyTorch tensor in [0.0, 255.0].
    
    If pre_resize_size is specified (e.g. 128), the image is first resized to
    (pre_resize_size, pre_resize_size) via bilinear interpolation, and then
    resized to (image_size, image_size).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    if pre_resize_size is not None and pre_resize_size > 0:
        pre_transform = T.Resize((pre_resize_size, pre_resize_size), interpolation=T.InterpolationMode.BILINEAR)
        image = pre_transform(image)

    transforms = get_transforms(image_size=image_size, is_training=False)
    resized = transforms(image)
    
    np_img = np.array(resized, dtype=np.float32) # (H, W, 3)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1) # (3, H, W)
    return tensor_img


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (256, 256)
) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    return img_tensor


def extract_lota_forensic_features(
    raw_image: torch.Tensor,
    bit_indices: List[int] = [0, 1, 2],
    normalization: str = "thresholding",
    patch_size: int = 32,
    patch_strategy: str = "max_gradient"
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    noise_image = compose_low_bit_planes(raw_image, bit_indices=bit_indices)

    if normalization == "thresholding":
        norm_noise = normalize_noise_thresholding(noise_image)
    elif normalization == "scaling":
        norm_noise = normalize_noise_scaling(noise_image)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    selected_patch, selected_idx = maximum_gradient_patch_selection(
        norm_noise,
        patch_size=patch_size,
        strategy=patch_strategy
    )

    return norm_noise, selected_patch, int(selected_idx.item() if hasattr(selected_idx, 'item') else selected_idx)


def build_transforms(is_train: bool = True, image_size: int = 256):
    transform_list = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
    ]
    if is_train:
        transform_list.append(T.RandomHorizontalFlip(p=0.5))
    return T.Compose(transform_list)
