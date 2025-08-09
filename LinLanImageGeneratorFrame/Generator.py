from .struct import *
from .utils import *
from .attention import *


class Resnet2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resnet2dBlock, self).__init__()
        self.equal = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)
        )
        if not self.equal:
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return x + out


class GeneratorUp2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_resnet, attention: bool = False, head_num=8, d_model=64,
                 dropout=.1, dff=512, pos_embed=False,
                 image_size=32):
        super(GeneratorUp2dBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.resnet = nn.Sequential(
            *[Resnet2dBlock(in_channels=out_channels, out_channels=out_channels) for i in range(num_resnet)])
        self.have_attention = attention
        if self.have_attention:
            self.atten = ImageSelfAttentionBlock(in_channels=out_channels, num_head=head_num, d_model=d_model,
                                                 dropout=dropout, dff=dff, pos_embed=pos_embed,
                                                 image_size=image_size)
        self.image_size = image_size

    def forward(self, x):
        x = self.up(x)
        if self.have_attention:
            x = self.atten(x)
        x = self.resnet(x)
        return x


class GeneratorModel(nn.Module):
    def __init__(self, in_dim,
                 depth,
                 hidden_channels,
                 image_shape,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False):
        super(GeneratorModel, self).__init__()
        self.config = get_config(in_dim=in_dim, depth=depth, hidden_channels=hidden_channels, image_shape=image_shape,
                                 attention=attention, head_num=head_num, d_model=d_model, num_resnet=num_resnet,
                                 dropout=dropout, dff=dff, pos_embed=pos_embed, name=type(self).__name__)
        self.in_dim = in_dim
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.image_shape = image_shape
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Sequential(
            nn.Linear(in_dim, hidden_channels * (2 ** depth) * self.dim * self.dim),
            nn.Unflatten(1, (hidden_channels * (2 ** depth), self.dim, self.dim)),
        )
        self.set_layer(attention=attention,
                       head_num=head_num,
                       d_model=d_model,
                       num_resnet=num_resnet,
                       dropout=dropout,
                       dff=dff,
                       pos_embed=pos_embed,
                       in_size=image_shape[1])

    def set_layer(self, attention=False,
                  head_num=8,
                  d_model=64,
                  num_resnet=4,
                  dropout=.1,
                  dff=512,
                  pos_embed=False,
                  in_size=32):
        arr = []
        if not attention:
            attention = [False] * self.depth
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(GeneratorUp2dBlock(in_channels=self.hidden_channels * (2 ** i),
                                          out_channels=self.hidden_channels * (2 ** (i - 1)),
                                          attention=attention[index],
                                          head_num=head_num,
                                          d_model=d_model,
                                          num_resnet=num_resnet,
                                          dropout=dropout,
                                          dff=dff,
                                          pos_embed=pos_embed,
                                          image_size=in_size // (2 ** (i - 1))))
            index += 1
        self.layer = nn.Sequential(
            *arr,
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.image_shape[0], kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init(x)
        out = self.layer(x)
        return out

    @torch.no_grad()
    def sample(self, batch_size, device="cuda"):
        norise = torch.randn(size=(batch_size, self.in_dim)).to(device)
        out = self(norise)
        return out
