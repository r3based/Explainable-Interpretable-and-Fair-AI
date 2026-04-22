import torch
import torchvision.models as models


def load_model(device="cpu"):
    model = models.resnet18(pretrained=True)
    model.eval()
    model.to(device)
    return model