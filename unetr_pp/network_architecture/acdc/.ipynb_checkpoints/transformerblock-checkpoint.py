import torch.nn as nn
import torch
import torch.nn.functional as F
from unetr_pp.network_architecture.dynunet_block import UnetResBlock

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
            proj_size=proj_size, 
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
    """
    Efficient Channel-Depth attention block - Optimized for reduced FLOPs
    """
    def __init__(self, depth_size: int, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.proj_size = proj_size
        self.depth_size = depth_size
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        # Temperature parameters
        self.temperature = nn.Parameter(torch.ones(depth_size, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(hidden_size, 1, 1))
        
        # Linear layers for samentic information (channel wise attention) and for geometic information (depth attention)
        # self.qkv_c = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        # self.qkv_d = nn.Linear(depth_size, depth_size * 3, bias=qkv_bias)
        self.qkv_c = nn.Conv3d(hidden_size, hidden_size * 3, 
                       kernel_size=1, bias=qkv_bias)
        self.qkv_d = nn.Conv3d(depth_size, depth_size * 3, 
                       kernel_size=1, bias=qkv_bias)
        # Dropout layers
        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        
        # Output projections
        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    
    def forward(self, x):
        B, C, D, H, W = x.shape
        N = H * W
        
        # Channel attention using Conv3D
        # Apply conv3d directly on the 5D tensor
        qkv_c_conv = self.qkv_c(x)  # [B, 3*C, D, H, W]
        
        # Single reshape and permute operation
        qkv_c = qkv_c_conv.reshape(B, 3, C, D, N).permute(1, 0, 3, 4, 2)  # [3, B, D, N, C]
        q_c, k_c, v_c = qkv_c[0], qkv_c[1], qkv_c[2]  # Each: [B, D, N, C]
        
        # Transpose for attention computation
        q_c = q_c.transpose(-2, -1)  # [B, D, C, N]
        k_c = k_c.transpose(-2, -1)  # [B, D, C, N]
        v_c = v_c.transpose(-2, -1)  # [B, D, C, N]
        
        # Use inplace normalization to save memory
        q_c = F.normalize(q_c, dim=-1)
        k_c = F.normalize(k_c, dim=-1)
        
        # Compute channel attention
        attn_c = torch.matmul(q_c, k_c.transpose(-2, -1)) * self.temperature
        attn_c = F.softmax(attn_c, dim=-1)
        attn_c = self.attn_drop(attn_c)
        
        # Apply channel attention
        x_c = torch.matmul(attn_c, v_c).permute(0, 2, 3, 1).reshape(B, D, N, C)
        x_c = self.out_proj(x_c)
        
        # Depth attention using Conv3D
        # Permute to put depth in channel dimension for conv3d
        
        x_depth = x.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]
        qkv_d_conv = self.qkv_d(x_depth)  # [B, 3*D, C, H, W]
        # print(qkv_d_conv.shape)
        qkv_d_conv = qkv_d_conv.reshape(B, 3, D, C, H, W)  # [B, 3, D, C, H, W]
        
        # Reshape for attention computation - we want [3, B, C, N, D] for depth attention
        qkv_d_conv = qkv_d_conv.permute(1, 0, 3, 2, 4, 5)  # [3, B, C, D, H, W]
        qkv_d = qkv_d_conv.reshape(3, B, C, D, N).permute(0, 1, 2, 4, 3)  # [3, B, C, N, D]
        q_d, k_d, v_d = qkv_d[0], qkv_d[1], qkv_d[2]  # Each: [B, C, N, D]
        
        # Transpose for attention computation
        q_d = q_d.transpose(-2, -1)  # [B, C, D, N]
        k_d = k_d.transpose(-2, -1)  # [B, C, D, N]
        v_d = v_d.transpose(-2, -1)  # [B, C, D, N]
        
        # Use inplace normalization to save memory
        q_d = F.normalize(q_d, dim=-1)
        k_d = F.normalize(k_d, dim=-1)
        
        # Compute depth attention
        attn_d = torch.matmul(q_d, k_d.transpose(-2, -1)) * self.temperature2
        attn_d = F.softmax(attn_d, dim=-1)
        attn_d = self.attn_drop_2(attn_d)
        
        # Apply depth attention
        x_d = torch.matmul(attn_d, v_d).permute(0, 2, 3, 1).reshape(B, C, N, D)
        # Need to reshape to match x_c format [B, D, N, C]
        x_d = x_d.permute(0, 3, 2, 1).reshape(B, D, N, C)
        x_d = self.out_proj2(x_d)
        
        # Concatenate channel and depth features
        x_out = torch.cat((x_c, x_d), dim=-1)  # [B, D, N, C]
        
        # Reshape back to original format
        x_out = x_out.permute(0, 3, 1, 2).reshape(B, C, D, H, W)
        
        return x_out

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
