import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34

class CIFARNormalizedModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)

def _build_cifar_resnet(builder, num_classes):
    backbone = builder(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return CIFARNormalizedModel(backbone)

def get_resnet18(num_classes=10):
    return _build_cifar_resnet(resnet18, num_classes)

def get_resnet34(num_classes=2):
    return _build_cifar_resnet(resnet34, num_classes)
