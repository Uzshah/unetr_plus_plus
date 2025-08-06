import torch.nn as nn
import torch
import torch.nn.functional as F
from unetr_pp.network_architecture.dynunet_block import UnetResBlock
import math

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            depth_size = int,
            bottle_neck = False
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        # Use 3D normalization for 5D tensors (B, C, H, W, D)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        
        # Updated to use ECD instead of EPA
        # Note: depth_size parameter is derived from input_size assuming cubic volume
        # print(input_size)
        depth_size = depth_size
        # print(depth_size)
        self.ecd_block = ECD(
                depth_size=depth_size, 
                hidden_size=hidden_size, 
                # proj_size=proj_size, 
                num_heads=num_heads, 
                channel_attn_drop=dropout_rate,
                spatial_attn_drop=dropout_rate
            )
        
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        # Position embedding for 5D tensors (B, C, H, W, D)
        self.pos_embed = None
        if pos_embed:
            # Create learnable position embedding for each spatial location and channel
            self.pos_embed = nn.Parameter(torch.zeros(1, hidden_size, 1, 1, 1))

    def forward(self, x):
        # print(x.shape)
        B, C, D, H, W = x.shape

        # Add positional embedding if enabled
        if self.pos_embed is not None:
            x = x + self.pos_embed  # Broadcasting across H, W, D dimensions

        # Apply group normalization directly on 5D tensor
        x_normed = self.norm(x)
        
        # Apply ECD block (expects 5D input: B, C, H, W, D)
        attn_out = self.ecd_block(x_normed)
        
        # Apply gamma scaling with proper broadcasting
        # Reshape gamma to match the channel dimension
        gamma_reshaped = self.gamma.view(1, -1, 1, 1, 1)  # (1, C, 1, 1, 1)
        attn = x + gamma_reshaped * attn_out
        
        # Apply convolution blocks
        attn = self.conv51(attn)
        x = attn + self.conv8(attn)

        return x


class ECD(nn.Module):
    def __init__(self, depth_size: int, hidden_size: int, num_heads=4,
                 qkv_bias=False, spatial_attn_drop=0.1, channel_attn_drop=0.0):
        super().__init__()
        self.hidden_size = hidden_size  # C
        self.depth_size = depth_size    # D
        self.num_heads = num_heads
        self.scale = (hidden_size*depth_size // num_heads) ** -0.5

        # Linear: map C -> 3C
        self.to_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)

        # Output projection: must be C*D -> C*D, NOT C -> C
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.out_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop = nn.Dropout(spatial_attn_drop)

        # LayerNorm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = H * W  # Spatial positions

        # -------------------------------
        # 1. Reshape to (B, D, H, W, C)
        # -------------------------------
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, C, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, H, W, C)
        x = self.norm(x)

        # -------------------------------
        # 2. Linear: C -> 3C
        # -------------------------------
        qkv = self.to_qkv(x)  # (B, D, H, W, 3C)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, D, H, W, C)

        # -------------------------------
        # 3. Reshape to (B, H*W, C*D) for attention over spatial dim
        # -------------------------------
        # Permute: (B, D, H, W, C) -> (B, H, W, D, C) -> (B, H*W, D*C)
        q = q.permute(0, 2, 3, 1, 4).contiguous().view(B, N, D * C)
        k = k.permute(0, 2, 3, 1, 4).contiguous().view(B, N, D * C)
        v = v.permute(0, 2, 3, 1, 4).contiguous().view(B, N, D * C)

        # -------------------------------
        # 4. Multi-head attention over N = H*W
        # -------------------------------
        # Reshape for multi-head
        q = q.view(B, N, self.num_heads, -1).transpose(1, 2)  # (B, h, N, head_dim)
        k = k.view(B, N, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, N, self.num_heads, -1).transpose(1, 2)

        # Attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output
        out = torch.matmul(attn, v)  # (B, h, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, D, C)  # (B, N, C*D)

        # -------------------------------
        # 5. Project back: C*D -> C*D
        # -------------------------------
        out = self.proj_out(out)  # (B, N, C*D)
        out = self.out_drop(out)

        # -------------------------------
        # 6. Reshape back to (B, C, D, H, W)
        # -------------------------------
        out = out.view(B, H, W, D, C)
        out = out.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, D, H, W)

        return out