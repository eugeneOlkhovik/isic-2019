
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


def get_EffNet(cfg, name='efficientnet-b3'):

    model = EfficientNet.from_pretrained(name,
                                         num_classes=cfg.num_classes)

    # # freeze, such the gradients are not computed in backward()
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace original fc with task-specific

    return model.to(cfg.DEVICE)


if __name__ == '__main__':

    class Config():
        num_classes = 6
        DEVICE = 'cuda'

    cfg = Config()
    model = get_EffNet(cfg)
