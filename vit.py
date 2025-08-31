import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class PatchEmbedding(nn.Module):
    """
    将输入图像分成 patch 并映射到 embedding 空间
    """
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, emb_size, H/patch, W/patch]
        x = self.proj(x)
        x = x.flatten(2)          # [B, emb_size, N_patches]
        x = x.transpose(1, 2)     # [B, N_patches, emb_size]
        return x

class Attention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, emb_size=128, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.head_dim = emb_size // num_heads

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*emb_size]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [B, heads, N, head_dim]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.fc_out(out)
        out = self.dropout(out)
        return out

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block: LayerNorm + Multi-head Attention + MLP + Residual
    """
    def __init__(self, emb_size=128, num_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = Attention(emb_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size*mlp_ratio),
            nn.GELU(),
            nn.Linear(emb_size*mlp_ratio, emb_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """
    简单的 Vision Transformer
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 emb_size=128, depth=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, N_patches, emb_size]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, 1+N, emb_size]
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 取 cls token
        out = self.head(cls_token_final)
        return out




def testNet(workNet):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU
    model = workNet.to(DEVICE)

    x = torch.randn(1,3,32,32).to(DEVICE)
    y = model(x)
    print(y.size())

    summary(model,(3,32,32))

if __name__ == "__main__":
    model = ViT(img_size=32, patch_size=4, emb_size=128, depth=6)
    testNet(model)
    exit
