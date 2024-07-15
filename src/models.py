import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),    # (b, c, t) -> (b, 128, t)
            ConvBlock(hid_dim, hid_dim),    # (b, 128, t) -> (b, 128, t)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # (b, 128, t) -> (b, 128, 1)
            Rearrange("b d 1 -> b d"),  # (b, 128, 1) -> (b, 128)
            nn.Linear(hid_dim, num_classes),    # (b, 128) -> (b, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b(batch_size), c(in_channels), t(seq_len) ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)  # (b, c, t) -> (b, 128, t)

        return self.head(X) # (b, 128, t) -> (b, num_classes)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)