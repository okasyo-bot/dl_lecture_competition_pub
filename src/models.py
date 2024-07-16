import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# ------------------
#    Baseline
# ------------------
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


# ------------------
#    Original
# ------------------
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim)    # (b, c, t) -> (b, 128, t)
        )


        self.head = nn.Sequential(
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            nn.AdaptiveAvgPool1d(1),    # (b, 128, t) -> (b, 128, 1)
            Rearrange("b d 1 -> b d"),  # (b, 128, 1) -> (b, 128)
            nn.Linear(hid_dim, num_classes),    # (b, 128) -> (b, num_classes)
        )

        self.attention = OriginalMultiheadAttention(in_dim=hid_dim, out_dim=hid_dim)

    def forward(self, X: torch.Tensor, subject_idxs) -> torch.Tensor:
        """_summary_
        Args:
            X ( b(batch_size), c(in_channels), t(seq_len) ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        #print(X.shape)
        X = self.blocks(X)  # (b, c, t) -> (b, 128, t)
        X = self.attention(X)   # (b, 128, t) -> (b, 128, t)

        #被験者情報の利用
        idxs = torch.zeros(X.shape[0], X.shape[1], 4).to(X.device)
        for i in range(X.shape[0]):
            idx = subject_idxs[i]
            for j in range(X.shape[1]):
                idxs[i][j][idx] = 1
        X = torch.cat((X, idxs), 2) # (b, 128, t) -> (b, 128, t+4)
        #print(X.shape)
        
        return self.head(X) # (b, 128, t+4) -> (b, num_classes)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        #self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
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
        #X = self.dropout(X)

        #X = self.conv2(X)
        #X = F.glu(X, dim=-2)

        return self.dropout(X)


class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, dim_head=64, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim_head = dim_head

        inner_dim = dim_head * heads
        self.scaler = self.head_dim ** (1/2)

        self.q = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.k = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.v = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.output = nn.Linear(self.in_dim, self.out_dim)
        #self.layer_norm = nn.LayerNorm(in_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        b, c, t = X.shape
        #print(X.shape)

        X = X.permute(0, 2, 1)
        #print(X.shape)

        q = self.q(X)
        k = self.k(X)
        v = self.v(X)

        # Split the embedding into self.heads different pieces
        query = q.reshape(b, t, self.heads, self.head_dim)
        key = k.reshape(b, t, self.heads, self.head_dim)
        value = v.reshape(b, t, self.heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        dot = torch.einsum("bhqd, bhkd -> bhqk", query, key)
        attention = torch.softmax(dot / self.scaler, dim=-1)

        out = torch.einsum("bhqk, bhvd -> bhqd", attention, value)
        out = out.contiguous().view(b, t, self.heads * self.head_dim)

        # 最終的な線形層
        out = self.output(out)
        out = self.dropout(out)

        # 元の形状 (batch_size, num_channels, seq_length) に戻す
        out = out.permute(0, 2, 1)

        return out



class OriginalMultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = self.in_dim // self.heads
        self.scaler = self.head_dim ** (1/2)

        self.q = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.k = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.v = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.output = nn.Linear(self.in_dim, self.out_dim)
        #self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        b, c, t = X.shape
        #print(X.shape)

        X = X.permute(0, 2, 1)
        #print(X.shape)

        q = self.q(X)
        k = self.k(X)
        v = self.v(X)

        # Split the embedding into self.heads different pieces
        query = q.reshape(b, t, self.heads, self.head_dim)
        key = k.reshape(b, t, self.heads, self.head_dim)
        value = v.reshape(b, t, self.heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        dot = torch.einsum("bhqd, bhkd -> bhqk", query, key)
        attention = torch.softmax(dot / self.scaler, dim=-1)

        out = torch.einsum("bhqk, bhvd -> bhqd", attention, value)
        out = out.contiguous().view(b, t, self.heads * self.head_dim)

        # 最終的な線形層
        out = self.output(out)
        out = self.dropout(out)

        # 元の形状 (batch_size, num_channels, seq_length) に戻す
        out = out.permute(0, 2, 1)

        return out
