from torchvision import models
import torch.nn as nn


def get_ResNet(cfg):

    model = models.resnet34(pretrained=True)

    # # freeze, such the gradients are not computed in backward()
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace original fc with task-specific
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.num_classes)

    return model.to(cfg.DEVICE)