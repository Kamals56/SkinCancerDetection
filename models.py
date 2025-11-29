import torch
import torch.nn as nn
import torchvision.models as models

def get_model(backbone="resnet50", num_classes=7, pretrained=True):
    """
    Returns a PyTorch ResNet model for skin lesion classification.

    Args:
        backbone (str): 'resnet18', 'resnet34', or 'resnet50'
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained ImageNet weights

    Returns:
        model (nn.Module): PyTorch model
    """
    backbone = backbone.lower()

    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Replaces the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model