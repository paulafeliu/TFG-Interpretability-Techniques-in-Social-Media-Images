"""
multitask_model.py
------------------
Defines the MultiTaskModel which uses a backbone and has multiple output heads.
"""
import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, feature_dim):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.fc_nature = nn.Linear(feature_dim, 2)
        self.fc_materiality = nn.Linear(feature_dim, 3)
        self.fc_biological = nn.Linear(feature_dim, 3)
        self.fc_landscape = nn.Linear(feature_dim, 8)

    def forward(self, x):
        features = self.backbone(x)
        out_nature = self.fc_nature(features)
        out_materiality = self.fc_materiality(features)
        out_biological = self.fc_biological(features)
        out_landscape = self.fc_landscape(features)
        return out_nature, out_materiality, out_biological, out_landscape
