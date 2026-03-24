"""
简单 CNN 分类模型

直接在低分辨率 2D 质谱图像上做样本分类，无需抽峰。
适配质谱图像特点：大尺寸、极度稀疏、变尺寸输入。

使用 AdaptiveAvgPool 处理任意输入尺寸，
log1p + BatchNorm 处理稀疏性和强度动态范围。
"""
import torch
import torch.nn as nn


class MzrtCNN(nn.Module):
    """轻量级 CNN 分类器

    架构：3 层卷积 + 全局自适应池化 + 全连接
    - 通道数少（8→16→32），匹配稀疏质谱图像的低信息密度
    - AdaptiveAvgPool2d(1,1) 全局平均池化，接受任意尺寸输入
    - Dropout 防止小样本过拟合
    - 总参数量 ~6K（适合几十到几百样本的典型代谢组学数据集）

    Args:
        num_classes: 分类类别数
        in_channels: 输入通道数（默认 1，单通道灰度图）
        base_filters: 第一层卷积核数（后续逐层翻倍）
        dropout: Dropout 比例
    """

    def __init__(self, num_classes, in_channels=1, base_filters=8, dropout=0.5):
        super().__init__()
        f = base_filters

        self.features = nn.Sequential(
            # Block 1: (1, H, W) → (f, H/2, W/2)
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (f, H/2, W/2) → (2f, H/4, W/4)
            nn.Conv2d(f, f * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (2f, H/4, W/4) → (4f, H/8, W/8)
            nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),

            # 全局池化：任意尺寸 → (4f, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(f * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """提取卷积特征向量（用于下游分析，如 t-SNE、聚类）

        Returns:
            (4f,) 维特征向量
        """
        x = self.features(x)
        return x.view(x.size(0), -1)
