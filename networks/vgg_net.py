import torch
import torch.nn as nn

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
         512, 512, 512, "M", 512, 512, 512, "M"]

# image_leftover = in / (2 ** VGG16.count('M'))


class VGGNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     ),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGNet(in_channels=3, num_classes=10).to(device)
    print(model)

    N = 3 # Mini batch size
    x = torch.randn(N, 3, 224, 224).to(device)
    print(model(x))
