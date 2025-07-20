import torch.nn as nn
import torch
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
                 channel_attn_drop=0.1, spatial_attn_drop=0.1, 
                 noise_std=0.01, noise_schedule='fixed', apply_noise_to='input'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size//num_heads
        
        # Noise parameters
        self.noise_std = noise_std
        self.noise_schedule = noise_schedule  # 'fixed', 'decay', 'adaptive'
        self.apply_noise_to = apply_noise_to  # 'input', 'attention', 'features', 'all'
        self.training_step = 0
        
        # Original layers
        self.qkv_c = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.qkv_d = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))
    
    def _get_noise_std(self):
        """Get noise standard deviation based on schedule"""
        if self.noise_schedule == 'fixed':
            return self.noise_std
        elif self.noise_schedule == 'decay':
            # Exponential decay: starts high, decreases over time
            decay_rate = 0.99
            return self.noise_std * (decay_rate ** self.training_step)
        elif self.noise_schedule == 'adaptive':
            # Adaptive based on training progress (could be customized)
            return self.noise_std * max(0.1, 1.0 - self.training_step / 10000)
        else:
            return self.noise_std
    
    def _add_gaussian_noise(self, tensor, std=None):
        """Add Gaussian noise to tensor during training"""
        if not self.training:
            return tensor
        
        if std is None:
            std = self._get_noise_std()
        
        noise = torch.randn_like(tensor) * std
        return tensor + noise
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        N = H * W
        
        # Update training step for noise scheduling
        if self.training:
            self.training_step += 1
        
        # Apply noise to input if specified
        if self.apply_noise_to in ['input', 'all']:
            x = self._add_gaussian_noise(x)
        
        x_reshaped = x.permute(0, 2, 3, 4, 1).view(B, D, N, C)
        
        qkv_c = self.qkv_c(x_reshaped).reshape(B, D, N, 3, C).permute(3, 0, 1, 2, 4)
        q_d = self.qkv_d(x_reshaped).reshape(B, D, N, C)
        q_c, k_c, v_c = qkv_c[0], qkv_c[1], qkv_c[2]
        k_shared, v_shared = k_c, v_c
        
        # Apply noise to features if specified
        if self.apply_noise_to in ['features', 'all']:
            q_c = self._add_gaussian_noise(q_c, std=self._get_noise_std() * 0.5)
            k_c = self._add_gaussian_noise(k_c, std=self._get_noise_std() * 0.5)
            v_c = self._add_gaussian_noise(v_c, std=self._get_noise_std() * 0.5)
            q_d = self._add_gaussian_noise(q_d, std=self._get_noise_std() * 0.5)
        
        # Channel attention 
        q_c = q_c.reshape(B*D, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        k_c = k_c.reshape(B*D, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        v_c = v_c.reshape(B*D, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        
        q_c = q_c.transpose(-2, -1)
        k_c = k_c.transpose(-2, -1)
        v_c = v_c.transpose(-2, -1)
        q_c = F.normalize(q_c, dim=-1)
        k_c = F.normalize(k_c, dim=-1)
        
        attn_CA = (q_c @ k_c.transpose(-2, -1)) / (N ** 0.5)
        
        # Apply noise to attention weights if specified
        if self.apply_noise_to in ['attention', 'all']:
            attn_CA = self._add_gaussian_noise(attn_CA, std=self._get_noise_std() * 0.1)
        
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_c).reshape(B, D, C, N).permute(0, 1, 3, 2)
        x_CA = self.out_proj(x_CA)
        
        # Depth attention
        q_d = q_d.permute(0, 3, 2, 1).reshape(B*C, N, D) 
        k_d = k_shared.permute(0, 3, 2, 1).reshape(B*C, N, D) 
        v_d = v_shared.permute(0, 3, 2, 1).reshape(B*C, N, D) 
        
        q_d = q_d.transpose(-2, -1)
        k_d = k_d.transpose(-2, -1)
        v_d = v_d.transpose(-2, -1)
        q_d = F.normalize(q_d, dim=-1)
        k_d = F.normalize(k_d, dim=-1)
        
        attn_D = (q_d @ k_d.transpose(-2, -1)) / (N ** 0.5)
        
        # Apply noise to depth attention weights if specified
        if self.apply_noise_to in ['attention', 'all']:
            attn_D = self._add_gaussian_noise(attn_D, std=self._get_noise_std() * 0.1)
        
        attn_D = attn_D.softmax(dim=-1)
        attn_D = self.attn_drop_2(attn_D)
        x_D = (attn_D @ v_d)
        x_D = x_D.reshape(B, C, D, N).permute(0, 2, 3, 1)
        x_D = self.out_proj2(x_D)
        
        x = torch.cat((x_CA, x_D), dim=-1).permute(0, 3, 1, 2).reshape(B, C, D, H, W)
        
        return x