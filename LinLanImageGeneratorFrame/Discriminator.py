from .globals import *
from .utils import *
from .struct import *

class DiscriminatorDown2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spectral_norm_flag=False):
        super(DiscriminatorDown2dBlock, self).__init__()
        if spectral_norm_flag:
            self.layer = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            )

    def forward(self, x):
        out = self.layer(x)
        return out

class DiscriminatorModel(nn.Module):
    def __init__(self, image_shape, hidden_channels, depth):
        super(DiscriminatorModel, self).__init__()
        self.config = get_config(image_shape=image_shape, hidden_channels=hidden_channels, depth=depth,
                                 name=type(self).__name__)
        self.image_shape = image_shape
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.dim = image_shape[1] // (2 ** depth)
        self.set_layer()

    def set_layer(self):
        arr = [nn.Conv2d(in_channels=self.image_shape[0], out_channels=self.hidden_channels, kernel_size=1)]
        for i in range(self.depth):
            arr.append(DiscriminatorDown2dBlock(
                in_channels=self.hidden_channels * (2 ** i),
                out_channels=self.hidden_channels * (2 ** (i + 1)),
                spectral_norm_flag=True
            ))
        self.layer = nn.Sequential(
            *arr,
            nn.Flatten(1),
            nn.Linear(self.hidden_channels * (2 ** self.depth) * self.dim * self.dim, 1),
        )

    def forward(self, x):
        out = self.layer(x)
        return out
