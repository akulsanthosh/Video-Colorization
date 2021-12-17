import torch.nn as nn


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, ):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=nn.BatchNorm2d),
                                 nn.BatchNorm2d(ndf * 2),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=nn.BatchNorm2d))

    def forward(self, input):
        return self.net(input)
