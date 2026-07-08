"""
CNN Backbone 网络
==================
提供 Conv-4 和 ResNet-12 两种特征提取器。
Prototypical Networks 论文中使用 Conv-4（4层卷积）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Conv-4（推荐基线）───
class Conv4(nn.Module):
    """4层卷积网络，与 Prototypical Networks 论文一致

    输入:  (B, 3, 84, 84)
    输出:  (B, 64)  —— 64维嵌入向量
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64, z_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 3×84×84 → 64×42×42
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64×42×42 → 64×21×21
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64×21×21 → 64×10×10
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 64×10×10 → 64×5×5
            nn.Conv2d(hidden, z_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 全局平均池化: 64×5×5 → 64
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 84, 84) 图像 tensor
        Returns:
            (B, 64) 嵌入向量
        """
        x = self.encoder(x)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)


# ─── ResNet-12（更强，可选扩展）───
class ResidualBlock(nn.Module):
    """残差块: 3×3 conv → BN → ReLU → 3×3 conv → BN → 残差连接 → ReLU"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet12(nn.Module):
    """ResNet-12: 4个 block，每 block 包含若干残差块

    输入:  (B, 3, 84, 84)
    输出:  (B, 512) —— 512维嵌入
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64, z_dim: int = 512):
        super().__init__()
        self.in_planes = hidden

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # Res blocks
        self.layer1 = self._make_layer(hidden,    1, stride=1)   # 64×84×84
        self.layer2 = self._make_layer(hidden*2,  1, stride=2)   # 128×42×42
        self.layer3 = self._make_layer(hidden*4,  1, stride=2)   # 256×21×21
        self.layer4 = self._make_layer(hidden*8,  1, stride=2)   # 512×10×10

        # 输出投影
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden * 8, z_dim)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        layers = [ResidualBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.proj(x)


# ─── 工厂函数 ───
def get_backbone(name: str = 'conv4', **kwargs):
    """获取 backbone

    Args:
        name: 'conv4' | 'resnet12'
    """
    if name == 'conv4':
        return Conv4(**kwargs)
    elif name == 'resnet12':
        return ResNet12(**kwargs)
    else:
        raise ValueError(f"未知 backbone: {name}")


if __name__ == '__main__':
    # 快速测试
    x = torch.randn(4, 3, 84, 84)

    conv4 = Conv4()
    print(f"Conv4 output shape: {conv4(x).shape}")    # (4, 64)

    res12 = ResNet12()
    print(f"ResNet12 output shape: {res12(x).shape}")  # (4, 512)
