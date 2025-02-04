import torch
from PIL import Image
from torchvision.transforms import ToTensor as ImageToTensorTransform
from typing import List


image_to_tensor_transform = ImageToTensorTransform()


def transform_images_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    Transforms a list of `PIL.Image.Image` images to a stacked tensor.

    Args:
        images: (List[Image.Image]) The list of images. There are `b`
            images, each with `c` channels, `h` pixels high, `w` pixels
            wide.

    Returns:
        torch.Tensor: The tensor of stacked images.
            Dim: `(b, c, h, w)`
    """
    return torch.cat(
        [image_to_tensor_transform(pic=image).unsqueeze(0) for image in images],
        dim=0,
    )
