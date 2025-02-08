import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        self.normalizer = Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]),
                                    std=torch.FloatTensor([0.229, 0.224, 0.225]))

    def forward(self, obs, spacial=True, normalize=True):
        obs = obs[:, -3:] / 255.0 
        h = self.normalizer(obs)
        i = 0
        for m in list(self.encoder.children()):
            i += 1
            if i <= 8:
                h = m(h)
        h = h.view(obs.shape[0], -1)
        return h
