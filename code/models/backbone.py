"""
backbone.py
-----------
Contains the CustomBackbone class for selecting a pretrained feature extractor.
"""
import torch.nn as nn
import torchvision.models as models


class CustomBackbone(nn.Module):
    def __init__(self, model_choice='ResNet18'):
        super(CustomBackbone, self).__init__()
        self.model_choice = model_choice
        if model_choice == 'DenseNet121':
            model_base = models.densenet121(pretrained=True)
            model_base.classifier = nn.Identity()
            self.feature_dim = 1024
        elif model_choice == 'ResNet18':
            model_base = models.resnet18(pretrained=True)
            model_base.fc = nn.Identity()
            self.feature_dim = 512
        elif model_choice == 'EfficientNetB0':
            model_base = models.efficientnet_b0(pretrained=True)
            model_base.classifier = nn.Identity()
            self.feature_dim = 1280
        else:
            model_base = models.resnet50(pretrained=True)
            model_base.fc = nn.Identity()
            self.feature_dim = 2048
        self.backbone = model_base

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)  # Flatten features
