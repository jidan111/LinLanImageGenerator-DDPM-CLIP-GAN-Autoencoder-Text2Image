from .globals import *
from .utils import *


def channels2group_nums(channels):
    if channels < 8:
        return 0
    arr = [64, 32, 24, 16, 8]
    for i in arr:
        if channels % i == 0:
            return channels // i
    return 1


def get_depth(size, min_size=4):
    n = 0
    while size > min_size and size % 2 == 0:
        size //= 2
        n += 1
    return n, size


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(ResnetBlock, self).__init__()
        self.equal = in_channels == out_channels
        group_nums = channels2group_nums(out_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.GroupNorm(num_channels=out_channels, num_groups=group_nums) if group_nums > 1 else nn.InstanceNorm2d(
                out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        if not self.equal:
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return self.activation(x + out)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, num_resnet=2, kernel_size=7):
        super(DownSample, self).__init__()
        self.resnet = nn.Sequential(
            *[ResnetBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
              for _ in range(num_resnet)])
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.resnet(x)
        out = self.down(x)
        return out


class Conv2d_Features(nn.Module):
    def __init__(self, image_shape, min_depth=2, hidden_channels=16, out_dim=512, min_size=8, kernel_size=7):
        super(Conv2d_Features, self).__init__()
        assert image_shape[1] == image_shape[2], f"只接收输入方形图片, {image_shape[1]}!={image_shape[2]}"
        self.config = get_config(image_shape=image_shape, min_depth=min_depth, hidden_channels=hidden_channels,
                                 out_dim=out_dim, min_size=min_size, kernel_size=kernel_size)
        self.depth, self.dim = get_depth(size=image_shape[1], min_size=min_size)
        self.hidden_channels = hidden_channels
        assert self.depth >= min_depth, f"卷积深度只有{self.depth}层，建议使用更大的图片"
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1),
            *[DownSample(in_channels=hidden_channels * (2 ** i), out_channels=hidden_channels * (2 ** (i + 1)),
                         kernel_size=kernel_size)
              for i in range(self.depth)]
        )
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels * (2 ** self.depth), out_channels=out_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        batch_size, *_ = x.shape
        x = self.encoder(x)
        out = self.final(x)
        return out.squeeze()
