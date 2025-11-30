import torch
import torch.nn as nn
import torchvision.models as models

def get_model(backbone="resnet50", num_classes=7, pretrained=True, dropout=0.5, hidden_units=512):
    """
    Returns a PyTorch ResNet model for skin lesion classification with optional
    hidden layer and dropout.

    Args:
        backbone (str): 'resnet18', 'resnet34', or 'resnet50'
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained ImageNet weights
        dropout (float): Dropout rate before final layer
        hidden_units (int): Number of units in the optional hidden FC layer

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

    in_features = model.fc.in_features

    # Replace final layer with a hidden layer + dropout + output layer
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, num_classes)
    )

    return model
