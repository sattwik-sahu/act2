import torch
from abc import abstractmethod, ABC
from PIL import Image
from typing import List
from torchvision.models import resnet
from torchvision.transforms import ToTensor
from typing_extensions import override
from helpers import DEVICE
from helpers.image import transform_images_to_tensor


torch.set_default_device(device=DEVICE)


class ImageEncoderBackbone(torch.nn.Module, ABC):
    """
    Abstract class for an image encoding backbone.
    """

    tensor_transform: ToTensor

    def __init__(self) -> None:
        torch.nn.Module.__init__(self=self)
        self.tensor_transform = ToTensor()

    def _preprocess(self, images: torch.Tensor | List[Image.Image]) -> torch.Tensor:
        """
        Converts the image list into a tensor as required in `self.forward`,
        if in the `List[PIL.Image.Image]` format else passes through as it is.
        If `images` is a `torch.Tensor`, check if it is 4-dimensional.

        Args:
            images (torch.Tensor | List[PIL.Image.Image]):
                The images to be encoded.

        Returns:
            torch.Tensor: The encoded images. Dim: `(b, ...)`
        """
        if type(images) is torch.Tensor:
            # Has 4 dimensions?
            assert len(images.shape) == 4, (
                f"Expected `images` to have 4 dimensions, got {len(images.shape)}"
            )
            return images
        elif type(images) is list:
            # Convert each image to a tensor and stack
            return transform_images_to_tensor(images=images).to(device=DEVICE)
        else:
            # Return empty
            raise Exception(
                "Type of `images` should be `torch.Tensor | List[Image.Image]`"
            )

    @abstractmethod
    def forward(self, images: torch.Tensor | List[Image.Image]) -> torch.Tensor:
        """
        Takes in images and encodes it using the encoder backbone.

        Args:
            images (torch.Tensor | List[PIL.Image.Image]):
                The images to be encoded. Dim: `(b, c, h, w)` or a list of
                `PIL.Image.Image`.

        Returns:
            torch.Tensor: The encoded images. Dim: `(b, ...)`
        """
        pass


class Resnet18Backbone(ImageEncoderBackbone):
    """
    The ResNet18 backbone for image encoding.
    """

    resnet18: torch.nn.Sequential
    OUTPUT_DIM: int = 512

    def __init__(self) -> None:
        torch.nn.Module.__init__(self=self)
        resnet18: resnet.ResNet = resnet.resnet18(
            weights=resnet.ResNet18_Weights.DEFAULT
        ).to(device=DEVICE)
        self.resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-2])

    @override
    def forward(self, images: torch.Tensor | List[Image.Image]) -> torch.Tensor:
        images = self._preprocess(images=images)
        return self.resnet18(images).permute(0, 2, 3, 1).flatten(1, 2)  # Dim: (b, ...)


def main():
    backbone = Resnet18Backbone()
    pil_images = [
        Image.open(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-12-10__hound_hillside/frames/video_005/frame_000007.png"
        ),
        Image.open(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-12-10__hound_hillside/frames/video_005/frame_000016.png"
        ),
    ]
    tensor_images = torch.randn((5, 3, 640, 480))
    print(f"Shape of output for `tensor_images`: {backbone(tensor_images).shape}")
    print(f"Shape of output for `pil_images`: {backbone(pil_images).shape}")


if __name__ == "__main__":
    main()
