from typing import List, Tuple

import torch
from PIL import Image
from torch.nn.modules.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings

from helpers import DEVICE
from utils.models.image_encoder import ImageEncoderBackbone, Resnet18Backbone

from utils.models.stylevar import StyleVarModule

torch.set_default_device(device=DEVICE)


class ActionTransformerEncoder(torch.nn.Module):
    image_enc_backbone: ImageEncoderBackbone
    image_embedding_layer: torch.nn.Linear
    joints_embedding_layer: torch.nn.Linear
    z_layer: torch.nn.Linear
    image_rope: RotaryPositionalEmbeddings
    encoder: TransformerEncoder

    def __init__(
        self,
        embed_dim: int = 512,
        n_joints: int = 14,
        n_heads: int = 8,
        n_layers: int = 4,
        z_dim: int = 32,
        image_enc_backbone: ImageEncoderBackbone = Resnet18Backbone(),
    ) -> None:
        torch.nn.Module.__init__(self=self)

        # Image encoding backbone
        self.image_enc_backbone = image_enc_backbone

        # Get image encoding backbone output size by passing dummy input
        image_enc_output_size: int = image_enc_backbone(
            torch.randn((1, 3, 480, 640))
        ).shape[-1]

        # Create image embedding layer
        self.image_embedding_layer = torch.nn.Linear(
            in_features=image_enc_output_size, out_features=embed_dim
        )

        # Image positional encodings
        self.image_rope = RotaryPositionalEmbeddings(dim=embed_dim)

        # Joints embedding layer
        self.joints_embedding_layer = torch.nn.Linear(
            in_features=n_joints, out_features=embed_dim
        )

        # Style variable linear layer
        self.z_layer = torch.nn.Linear(in_features=z_dim, out_features=embed_dim)

        # The transformer encoder
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, batch_first=True
            ),
            num_layers=n_layers,
        )

    def _embed_images(self, images: List[Image.Image] | torch.Tensor) -> torch.Tensor:
        # Dim: (n_images, ?, bb_output_dim)
        image_enc: torch.Tensor = self.image_enc_backbone(images)

        # Pass through linear layer
        # Dim: (n_images, ?, embed_dim)
        image_enc = self.image_embedding_layer(image_enc)

        # Perform positional encoding individually on each image encoding
        # Dim: (n_images, ?, 1, embed_dim)
        image_enc_pe: torch.Tensor = self.image_rope(image_enc.unsqueeze(2)).squeeze(2)

        # Stack image embeddings together
        # Dim: (n_images * ?, embed_dim)
        image_embeddings_stacked: torch.Tensor = torch.cat(tuple(image_enc_pe), dim=0)

        return image_embeddings_stacked

    def forward(
        self,
        images: List[Image.Image] | torch.Tensor,
        joints_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        image_embeddings: torch.Tensor = self._embed_images(images=images)

        # Dim: (1, embed_dim)
        joints_embedding: torch.Tensor = self.joints_embedding_layer(joints_obs)

        # Dim: (1, embed_dim)
        z_embedding: torch.Tensor = self.z_layer(z)
        return self.encoder(
            torch.cat((image_embeddings, joints_embedding, z_embedding), dim=0)
        )


img = torch.randn((4, 3, 480, 640))
z = torch.randn((1, 1, 32))
joints = torch.randn((1, 1, 14))
actions = torch.randn((1, 5, 14))
encoder = ActionTransformerEncoder()
z = StyleVarModule()(actions, joints)

print(encoder(img, joints, z).shape)
