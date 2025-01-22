import torch
from torchtune.modules import RotaryPositionalEmbeddings
from helpers import DEVICE


torch.set_default_device(DEVICE)


class StyleVarModule(torch.nn.Module):
    """
    PyTorch module to implement the transformer encoder to output the
    style variable, `z`, described in the paper "Learning Fine-Grained
    Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)[1].

    Attributes:
        embed_dim (int): The dimension of each of the embeddings.
        n_heads (int): Number of heads in the attention module of the
            transformer encoder.
        k (int): The variable, `k`, as described in [1].
        n_joints (int): The size of the action space.
        z_dim (int): Dimension of the style variable, `z`.
        cls_layer (torch.nn.Linear): The layer whose weights represent the
            learnable `[CLS]` token. Dim: `1 -> embed_dim`
        action_seq_emb_layer (torch.nn.Linear): The layer to embed the
            joint action sequence. Dim: `n_joints -> embed_dim`
        joints_emb_layer (torch.nn.Linear): The layer to embed the joint
            observations (angles). Dim: `n_joints -> embed_dim`
        sample_params_layer (torch.nn.Linear): The layer to estimate the
            sample parameters (mean and logvariance) of `z`.
            Dim: `embed_dim -> z_dim`
        rope (torchtune.modules.RotaryPositionalEmbeddings):
            The module to add rotary positional embeddings,described in the
            paper "RoFormer: Enhanced Transformer with Rotary Position
            Embedding" (Su et al., 2023)[2].
        encoder (torch.nn.TransformerEncoder): The transformer encoder module.
            Dim: `(b, (k + 2), embed_dim) -> (b, (k + 2), embed_dim)`, where
            `b` is the batch size.
    """

    embed_dim: int
    n_heads: int
    k: int
    n_joints: int
    z_dim: int
    cls_layer: torch.nn.Linear
    action_seq_emb_layer: torch.nn.Linear
    joints_embedding_layer: torch.nn.Linear
    sample_params_layer: torch.nn.Linear
    rope: RotaryPositionalEmbeddings
    encoder: torch.nn.TransformerEncoder

    def __init__(
        self,
        embed_dim: int = 512,
        n_heads: int = 8,
        k: int = 5,
        n_joints: int = 14,
        n_blocks: int = 4,
        z_dim: int = 32,
    ) -> None:
        """
        Initializes the encoder module to calculate the style variable, `z`.

        Args:
            embed_dim (int): The dimension of each of the embeddings.
            n_heads (int): Number of heads in the attention module of the
                transformer encoder.
            k (int): The variable, `k`, as described in [1].
            n_joints (int): The size of the action space.
            z_dim (int): Dimension of the style variable, `z`.
            n_blocks (int): The number of encoder layers in the encoder.
        """
        torch.nn.Module.__init__(self)

        # Module parameters
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.k = k
        self.n_joints = n_joints
        self.z_dim = z_dim

        # Initalize the required modules
        # Linear layers (1-4 in [1])
        self.cls_layer = torch.nn.Linear(in_features=1, out_features=embed_dim)
        self.joints_embbeding_layer = torch.nn.Linear(
            in_features=n_joints, out_features=embed_dim
        )
        self.action_emb_layer = torch.nn.Linear(
            in_features=n_joints, out_features=embed_dim
        )
        self.sample_params_layer = torch.nn.Linear(
            in_features=embed_dim, out_features=z_dim * 2
        )

        # Rotary positional embeddings, from [2]
        self.rope = RotaryPositionalEmbeddings(dim=embed_dim)

        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_blocks,
        )

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparametrization trick on using the given mean (`mu`) and
        log of variance (`logvar`).

        Args:
            mu (torch.Tensor): The mean of the sample.
            logvar (torch.Tensor): The log of the variance of the
                sample.

        Returns:
            torch.Tensor: A sample from the distribution described by the
                parameters `mu` and `logvar`
        """
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        z: torch.Tensor = mu + std * eps
        return z

    def forward(
        self, action_seq: torch.Tensor, joints_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Takes in an input sequence and outputs the style variable, `z`.

        Args:
            action_seq (torch.Tensor): The action sequence, containing `k`
                previous steps of actions (predicted joint space samples).
                Dim: `(b, k, n_joints)`
            joints_obs (torch.Tensor): The joint space observations.
                Dim: `(b, 1, n_joints)`

        Returns:
            torch.Tensor: The style variable, `z` [1]. Dim: `(b, z_dim)`

        Note:
            `b = 1` during inference time.
        """
        batch_size: int = action_seq.shape[0]

        # Verify input dimensions
        assert action_seq.shape == (batch_size, self.k, self.n_joints), (
            f"Expected action_seq of shape {(batch_size, self.k, self.n_joints)}, got {action_seq.shape}"
        )
        assert joints_obs.shape == (batch_size, 1, self.n_joints), (
            f"Expected joint_obs of shape {(batch_size, 1, self.n_joints)}, got {joints_obs.shape}"
        )

        # [CLS] token
        cls_input: torch.Tensor = torch.ones((batch_size, 1, 1))
        cls_token: torch.Tensor = self.cls_layer(cls_input)

        # Embedding the action sequence and adding rotary positional encodings
        action_seq_emb: torch.Tensor = self.action_emb_layer(action_seq)
        rope_action_seq_emb: torch.Tensor = self.rope(
            action_seq_emb.unsqueeze(0)
        ).squeeze(0)

        # Embed the joint observations
        joints_emb: torch.Tensor = self.joints_embbeding_layer(joints_obs)

        # Create the input for the encoder
        X_encoder: torch.Tensor = torch.cat(
            (
                cls_token,
                joints_emb,
                rope_action_seq_emb,
            ),
            dim=1,
        )

        # Extract CLS feature from encoder output
        cls_feature: torch.Tensor = self.encoder(X_encoder)[:, 0]

        # Get distribution parameters for style variable `z`
        sample_params: torch.Tensor = self.sample_params_layer(cls_feature)
        z_mean, z_logvar = sample_params[:, :32], sample_params[:, 32:]

        # Perform reparametrization and sample `z`
        z = self._reparametrize(mu=z_mean, logvar=z_logvar)

        return z


def main():
    # Declare parameters
    EMBED_DIM: int = 512
    N_HEADS: int = 8
    K: int = 5
    N_JOINTS: int = 14
    BATCH_SIZE: int = 100
    Z_DIM: int = 32

    # Init module
    z_module = StyleVarModule(
        embed_dim=EMBED_DIM, n_heads=N_HEADS, k=K, n_joints=N_JOINTS, z_dim=Z_DIM
    )

    # Construct example inputs
    action_seq = torch.randn((BATCH_SIZE, K, N_JOINTS))
    joints_obs = torch.randn((BATCH_SIZE, 1, N_JOINTS))

    print(
        f"Shapes of action sequence and joints observations: {action_seq.shape}, {joints_obs.shape}"
    )

    # Get the style variable
    z = z_module(action_seq, joints_obs)
    print(f"Shape of style variable `z`: {z.shape}")


if __name__ == "__main__":
    main()
