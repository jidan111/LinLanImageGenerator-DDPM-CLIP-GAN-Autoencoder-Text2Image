from .attention import *


def channels2group_nums(channels):
    if channels < 8:
        return 0
    arr = [64, 32, 24, 16, 8]
    for i in arr:
        if channels % i == 0:
            return channels // i
    return 1


class GateControlBlock(nn.Module):
    def __init__(self, x_channels, y_channels=None, out_channels=None):
        super(GateControlBlock, self).__init__()
        y_channels = y_channels if y_channels is not None else x_channels
        out_channels = out_channels if out_channels is not None else x_channels
        self.scale = nn.Sequential(
            nn.Conv2d(in_channels=x_channels + y_channels, out_channels=2, kernel_size=1, stride=1, bias=False),
            nn.Softmax(dim=1)
        )
        self.x_in = nn.Conv2d(in_channels=x_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.y_in = nn.Conv2d(in_channels=y_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, x, y):
        cat = torch.cat((x, y), dim=1)
        scale = self.scale(cat)
        x_weight, y_weight = scale.chunk(2, dim=1)
        x = self.x_in(x)
        y = self.y_in(y)
        out = x_weight * x + y_weight * y
        return out


class ConditionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConditionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
            nn.Linear(in_features=out_dim, out_features=out_dim)
        )
        self.out_dim = out_dim

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, self.out_dim, 1, 1)


class AdaGroupNormConditional(nn.Module):
    def __init__(self, in_channels, time_dim):
        super(AdaGroupNormConditional, self).__init__()
        self.condition_mlp = ConditionMLP(time_dim, in_channels * 2)
        num_groups = channels2group_nums(in_channels)
        if num_groups == 0:
            self.norm = nn.InstanceNorm2d(num_features=in_channels)
        else:
            self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups)

    def forward(self, x, time_embed):
        condition = self.condition_mlp(time_embed)
        scale, shift = condition.chunk(2, dim=1)
        out = self.norm(x)
        out = (1 + scale) * out + shift
        return out


class Resnet2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resnet2dBlock, self).__init__()
        self.equal = in_channels == out_channels
        num_groups = max(1, channels2group_nums(out_channels))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_channels=out_channels, num_groups=num_groups),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        if not self.equal:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                nn.GroupNorm(num_channels=out_channels, num_groups=num_groups)
            )
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return self.activation(x + out)


class Resnet2dBlockConditional(Resnet2dBlock):
    def __init__(self, in_channels, out_channels, condition_dim):
        super(Resnet2dBlockConditional, self).__init__(in_channels=in_channels, out_channels=out_channels)
        self.condition_mlp = ConditionMLP(in_dim=condition_dim, out_dim=in_channels)

    def forward(self, x, condition):
        condition = self.condition_mlp(condition)
        x = x + condition
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return self.activation(x + out)


class ImageSelfAttentionBlockConditional(ImageSelfAttentionBlock):
    def __init__(self, in_channels, condition_dim, num_head=8, d_model=64, dropout=.1, dff=512, pos_embed=False,
                 image_size=32):
        super(ImageSelfAttentionBlockConditional, self).__init__(in_channels=in_channels, num_head=num_head,
                                                                 d_model=d_model, dropout=dropout,
                                                                 dff=dff, pos_embed=pos_embed,
                                                                 image_size=image_size)
        self.image_size = image_size
        self.condition_mlp = ConditionMLP(in_dim=condition_dim, out_dim=in_channels)

    def forward(self, x, condition):
        batch_size, channels, h, w = x.shape
        x_in = x
        condition = self.condition_mlp(condition)
        x = x + condition
        x = self.norm1(x)
        x = self.init_conv(x)
        x = x.view(batch_size, self.hidden_dim, -1).transpose(1, 2).contiguous()
        if self.have_pos:
            x = x + self.pos_embed
        x = self.dropout1(self.self_atten(self.norm2(x)) + x)
        x = self.dropout2(self.ffc(self.norm3(x)) + x)
        x = x.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim, h, w)
        x = self.out_conv(x)
        return x + x_in


class Down2dBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False,
                 image_size=32):
        super(Down2dBlock, self).__init__()
        self.resnet = nn.Sequential(
            *[Resnet2dBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_resnet)])
        self.attention = attention
        if self.attention:
            self.atten = ImageSelfAttentionBlock(in_channels=in_channels, num_head=head_num, d_model=d_model,
                                                 dropout=dropout, dff=dff,
                                                 pos_embed=pos_embed,
                                                 image_size=image_size)
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        if self.attention:
            x = self.atten(x)
        x = self.resnet(x)
        out = self.down(x)
        return out


class Down2dBlockConditional(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 time_dim,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False,
                 image_size=32):
        super(Down2dBlockConditional, self).__init__()
        self.resnet = nn.ModuleList([Resnet2dBlockConditional(in_channels=in_channels, out_channels=in_channels,
                                                              condition_dim=time_dim) for i in range(num_resnet)])
        self.attention = attention
        if self.attention:
            self.atten = ImageSelfAttentionBlockConditional(in_channels=in_channels,
                                                            condition_dim=time_dim,
                                                            num_head=head_num,
                                                            d_model=d_model,
                                                            dropout=dropout,
                                                            dff=dff,
                                                            pos_embed=pos_embed,
                                                            image_size=image_size)
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = AdaGroupNormConditional(in_channels=in_channels, time_dim=time_dim)

    def forward(self, x, time_embed):
        if self.attention:
            x = self.atten(x, time_embed)
        for index in range(len(self.resnet)):
            x = self.resnet[index](x, time_embed)
        x = self.norm(x, time_embed)
        out = self.down(x)
        return out


class Down2dBlockConditionalWithTimeAndText(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, text_dim, head_num=8, num_resnet=4, d_model=64,
                 dropout=.1, dff=512, image_size=None, pos_embed=False, **kwargs):
        super(Down2dBlockConditionalWithTimeAndText, self).__init__()
        self.resnet = nn.ModuleList([Resnet2dBlockConditional(in_channels=in_channels, out_channels=in_channels,
                                                              condition_dim=time_dim) for i in range(num_resnet)])
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = AdaGroupNormConditional(in_channels=in_channels, time_dim=time_dim)
        self.corss_atten = ImageAndTextCrossAttentionBlock(in_channels=in_channels, context_dim=text_dim,
                                                           num_head=head_num, d_model=d_model, dropout=dropout, dff=dff,
                                                           image_size=image_size, pos_embed=pos_embed)

    def forward(self, x, time_embed=None, text_embed=None):
        for index in range(len(self.resnet)):
            x = self.resnet[index](x, time_embed)
        x = self.norm(x, time_embed)
        if text_embed is not None:
            x = self.corss_atten(x, text_embed)
        out = self.down(x)
        return out


class Up2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False,
                 image_size=32):
        super(Up2dBlock, self).__init__()
        self.resnet = nn.Sequential(
            *[Resnet2dBlock(in_channels=out_channels, out_channels=out_channels) for i in range(num_resnet)])
        self.attention = attention
        if self.attention:
            self.atten = ImageSelfAttentionBlock(in_channels=out_channels, num_head=head_num, d_model=d_model,
                                                 dropout=dropout, dff=dff,
                                                 pos_embed=pos_embed,
                                                 image_size=image_size)
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                     padding=1)

    def forward(self, x):
        out = self.up(x)
        if self.attention:
            out = self.atten(out)
        out = self.resnet(out)
        return out


class Up2dBlockConditional(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 time_dim,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False,
                 image_size=32):
        super(Up2dBlockConditional, self).__init__()
        self.resnet = nn.ModuleList([Resnet2dBlockConditional(in_channels=out_channels, out_channels=out_channels,
                                                              condition_dim=time_dim) for i in range(num_resnet)])
        self.attention = attention
        if self.attention:
            self.atten = ImageSelfAttentionBlockConditional(in_channels=out_channels,
                                                            condition_dim=time_dim,
                                                            num_head=head_num,
                                                            d_model=d_model,
                                                            dropout=dropout,
                                                            dff=dff,
                                                            pos_embed=pos_embed,
                                                            image_size=image_size)
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                     padding=1)
        self.norm = AdaGroupNormConditional(in_channels=in_channels, time_dim=time_dim)

    def forward(self, x, time_embed):
        out = self.norm(x, time_embed)
        out = self.up(out)
        if self.attention:
            out = self.atten(out, time_embed)
        for index in range(len(self.resnet)):
            out = self.resnet[index](out, time_embed)
        return out


class Up2dBlockConditionalWithTimeAndText(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, text_dim, head_num=8, num_resnet=4, d_model=64,
                 dropout=.1, dff=512, image_size=None, pos_embed=False, **kwargs):
        super(Up2dBlockConditionalWithTimeAndText, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet = nn.ModuleList([Resnet2dBlockConditional(in_channels=out_channels, out_channels=out_channels,
                                                              condition_dim=time_dim) for i in range(num_resnet)])
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                     padding=1)
        self.norm = AdaGroupNormConditional(in_channels=in_channels, time_dim=time_dim)
        self.cross_atten = ImageAndTextCrossAttentionBlock(in_channels=out_channels, context_dim=text_dim,
                                                           num_head=head_num, d_model=d_model, dropout=dropout, dff=dff,
                                                           pos_embed=pos_embed, image_size=image_size)

    def forward(self, x, time_embed, text_embed=None):
        x = self.norm(x, time_embed)
        out = self.up(x)
        if text_embed is not None:
            out = self.cross_atten(out, text_embed)
        for index in range(len(self.resnet)):
            out = self.resnet[index](out, time_embed)
        return out
