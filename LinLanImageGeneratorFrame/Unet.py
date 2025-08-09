from .struct import *
from .utils import get_config


class Unet2d(nn.Module):
    def __init__(self, image_shape,
                 depth,
                 hidden_channels,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False):
        super(Unet2d, self).__init__()
        assert image_shape[1] % 2 ** depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{2 ** depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.config = get_config(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels,
                                 attention=attention, head_num=head_num, d_model=d_model, num_resnet=num_resnet,
                                 dropout=dropout, dff=dff, pos_embed=pos_embed, name=type(self).__name__)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.set_encoder(attention=attention,
                         head_num=head_num,
                         d_model=d_model,
                         num_resnet=num_resnet,
                         dropout=dropout,
                         dff=dff,
                         pos_embed=pos_embed,
                         in_size=image_shape[1])
        self.set_decoder(attention=attention,
                         head_num=head_num,
                         d_model=d_model,
                         num_resnet=num_resnet,
                         dropout=dropout,
                         dff=dff,
                         pos_embed=pos_embed,
                         in_size=image_shape[1])
        self.set_midlayer(head_num=head_num,
                          d_model=d_model,
                          dropout=dropout,
                          dff=dff,
                          pos_embed=pos_embed,
                          in_size=image_shape[1] // (2 ** self.depth))
        self.set_merge_layer()
        self.out = nn.Sequential(
            Resnet2dBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=image_shape[0], kernel_size=1)
        )

    def set_merge_layer(self):
        arr = []
        for i in range(self.depth, 0, -1):
            arr.append(GateControlBlock(x_channels=self.hidden_channels * (2 ** i),
                                        y_channels=self.hidden_channels * (2 ** i),
                                        out_channels=self.hidden_channels * (2 ** i)
                                        ))
        self.merge_layer = nn.ModuleList(arr)

    def set_encoder(self, attention=False,
                    head_num=8,
                    d_model=64,
                    num_resnet=4,
                    dropout=.1,
                    dff=512,
                    pos_embed=False,
                    in_size=32):
        if not attention:
            attention = [False] * self.depth
        arr = []
        index = 0
        for h in range(self.depth):
            arr.append(Down2dBlock(in_channels=self.hidden_channels * (2 ** h),
                                   out_channels=self.hidden_channels * (2 ** (h + 1)),
                                   attention=attention[index],
                                   head_num=head_num,
                                   d_model=d_model,
                                   num_resnet=num_resnet,
                                   dropout=dropout,
                                   dff=dff,
                                   pos_embed=pos_embed,
                                   image_size=in_size // (2 ** index)))
            index += 1
        self.encoder = nn.ModuleList(arr)

    def set_decoder(self, attention=False,
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
        attention = attention[::-1]
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(Up2dBlock(in_channels=self.hidden_channels * (2 ** i),
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
        self.decoder = nn.ModuleList(arr)

    def set_midlayer(self, head_num=8,
                     d_model=64,
                     dropout=.1,
                     dff=512,
                     pos_embed=False,
                     in_size=32):
        self.mid_layer = nn.Sequential(
            Resnet2dBlock(in_channels=self.hidden_channels * (2 ** self.depth),
                          out_channels=self.hidden_channels * (2 ** self.depth)),
            ImageSelfAttentionBlock(in_channels=self.hidden_channels * (2 ** self.depth),
                                    num_head=head_num, d_model=d_model, dropout=dropout, dff=dff, pos_embed=pos_embed,
                                    image_size=in_size),
            Resnet2dBlock(in_channels=self.hidden_channels * (2 ** self.depth),
                          out_channels=self.hidden_channels * (2 ** self.depth)),
        )

    def forward(self, x):
        x = self.init(x)
        tmp = []
        for index, model in enumerate(self.encoder):
            if not tmp:
                tmp.append(model(x))
            else:
                tmp.append(model(tmp[-1]))
        mid = self.mid_layer(tmp.pop())
        out = None
        for index, model in enumerate(self.decoder):
            if out is None:
                out = model(mid)
            else:
                t = self.merge_layer[index](tmp.pop(), out)
                out = model(t)
        out = self.out(out)
        return out

    def get_decoder_last_layer(self):
        return self.out[1].weight


class Unet2dConditionalWithTime(nn.Module):
    def __init__(self, image_shape,
                 depth,
                 hidden_channels,
                 time_dim,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False):
        super(Unet2dConditionalWithTime, self).__init__()
        assert image_shape[1] % 2 ** depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.config = get_config(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels,
                                 time_dim=time_dim, attention=attention, head_num=head_num, d_model=d_model,
                                 num_resnet=num_resnet, dropout=dropout, dff=dff, pos_embed=pos_embed,
                                 name=type(self).__name__)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.init_condition = ConditionMLP(in_dim=time_dim, out_dim=hidden_channels)
        self.depth = depth
        self.time_dim = time_dim
        self.hidden_channels = hidden_channels
        self.set_encoder(attention=attention,
                         head_num=head_num,
                         d_model=d_model,
                         num_resnet=num_resnet,
                         dropout=dropout,
                         dff=dff,
                         pos_embed=pos_embed,
                         in_size=image_shape[1])
        self.set_decoder(attention=attention,
                         head_num=head_num,
                         d_model=d_model,
                         num_resnet=num_resnet,
                         dropout=dropout,
                         dff=dff,
                         pos_embed=pos_embed,
                         in_size=image_shape[1])
        self.set_midlayer(head_num=head_num,
                          d_model=d_model,
                          dropout=dropout,
                          dff=dff,
                          pos_embed=pos_embed,
                          in_size=image_shape[1] // (2 ** self.depth))
        self.set_merge_layer()
        self.out = nn.Sequential(
            Resnet2dBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=image_shape[0], kernel_size=1)
        )

    def set_merge_layer(self):
        arr = []
        for i in range(self.depth, 0, -1):
            arr.append(GateControlBlock(x_channels=self.hidden_channels * (2 ** i),
                                        y_channels=self.hidden_channels * (2 ** i),
                                        out_channels=self.hidden_channels * (2 ** i)
                                        ))
        self.merge_layer = nn.ModuleList(arr)

    def set_encoder(self, attention=False,
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
        for h in range(self.depth):
            arr.append(Down2dBlockConditional(in_channels=self.hidden_channels * (2 ** h),
                                              out_channels=self.hidden_channels * (2 ** (h + 1)),
                                              time_dim=self.time_dim,
                                              attention=attention[index],
                                              head_num=head_num,
                                              d_model=d_model,
                                              num_resnet=num_resnet,
                                              dropout=dropout,
                                              dff=dff,
                                              pos_embed=pos_embed,
                                              image_size=in_size // (2 ** index)))
            index += 1
        self.encoder = nn.ModuleList(arr)

    def set_decoder(self, attention=False,
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
        attention = attention[::-1]
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(Up2dBlockConditional(in_channels=self.hidden_channels * (2 ** i),
                                            out_channels=self.hidden_channels * (2 ** (i - 1)),
                                            time_dim=self.time_dim,
                                            attention=attention[index],
                                            head_num=head_num,
                                            d_model=d_model,
                                            num_resnet=num_resnet,
                                            dropout=dropout,
                                            dff=dff,
                                            pos_embed=pos_embed,
                                            image_size=in_size // (2 ** (i - 1))))
            index += 1
        self.decoder = nn.ModuleList(arr)

    def set_midlayer(self, head_num=8,
                     d_model=64,
                     dropout=.1,
                     dff=512,
                     pos_embed=False,
                     in_size=32):
        self.mid_layer = nn.ModuleList(
            [
                Resnet2dBlockConditional(in_channels=self.hidden_channels * (2 ** self.depth),
                                         out_channels=self.hidden_channels * (2 ** self.depth),
                                         condition_dim=self.time_dim),
                ImageSelfAttentionBlockConditional(in_channels=self.hidden_channels * (2 ** self.depth),
                                                   condition_dim=self.time_dim,
                                                   num_head=head_num, d_model=d_model,
                                                   dropout=dropout,
                                                   dff=dff,
                                                   pos_embed=pos_embed,
                                                   image_size=in_size),
                Resnet2dBlockConditional(in_channels=self.hidden_channels * (2 ** self.depth),
                                         out_channels=self.hidden_channels * (2 ** self.depth),
                                         condition_dim=self.time_dim)]
        )

    def forward(self, x, time_embed):
        x = self.init(x)
        condition_ = self.init_condition(time_embed)
        x = x + condition_
        tmp = []
        for index, model in enumerate(self.encoder):
            if not tmp:
                tmp.append(model(x, time_embed))
            else:
                tmp.append(model(tmp[-1], time_embed))
        mid = self.mid_layer[0](tmp.pop(), time_embed)
        mid = self.mid_layer[1](mid, time_embed)
        mid = self.mid_layer[2](mid, time_embed)
        out = None
        for index, model in enumerate(self.decoder):
            if out is None:
                out = model(mid, time_embed)
            else:
                t = self.merge_layer[index](tmp.pop(), out)
                out = model(t, time_embed)
        out = self.out(out)
        return out


class Unet2dConditionalWithTimeAndText(nn.Module):
    def __init__(self, image_shape, depth, hidden_channels, time_dim, text_dim, dropout=.1, head_num=8, d_model=64,
                 dff=512, num_resnet=4, pos_embed=True):
        super(Unet2dConditionalWithTimeAndText, self).__init__()
        assert image_shape[1] % 2 ** depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.config = get_config(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels,
                                 time_dim=time_dim, text_dim=text_dim, dropout=dropout, head_num=head_num,
                                 d_model=d_model, dff=dff, num_resnet=num_resnet, pos_embed=pos_embed,
                                 name=type(self).__name__)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.init_condition = ConditionMLP(in_dim=time_dim, out_dim=hidden_channels)
        self.depth = depth
        self.time_dim = time_dim
        self.text_dim = text_dim
        self.hidden_channels = hidden_channels
        self.set_encoder(num_resnet=num_resnet, dropout=dropout, head_num=head_num, d_model=d_model, dff=dff,
                         pos_embed=pos_embed, image_size=image_shape[1])
        self.set_decoder(num_resnet=num_resnet, dropout=dropout, head_num=head_num, d_model=d_model, dff=dff,
                         pos_embed=pos_embed, image_size=image_shape[1])
        self.set_midlayer(dropout=dropout, head_num=head_num, d_model=d_model, dff=dff, pos_embed=pos_embed,
                          image_size=image_shape[1] // (2 ** self.depth))
        self.set_merge_layer()
        self.out = nn.Sequential(
            Resnet2dBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=image_shape[0], kernel_size=1)
        )

    def set_merge_layer(self):
        arr = []
        for i in range(self.depth, 0, -1):
            arr.append(GateControlBlock(x_channels=self.hidden_channels * (2 ** i),
                                        y_channels=self.hidden_channels * (2 ** i),
                                        out_channels=self.hidden_channels * (2 ** i)
                                        ))
        self.merge_layer = nn.ModuleList(arr)

    def set_encoder(self, num_resnet=4, dropout=.1, head_num=8, d_model=64, dff=512, pos_embed=False, image_size=None):
        arr = []
        for h in range(self.depth):
            arr.append(Down2dBlockConditionalWithTimeAndText(in_channels=self.hidden_channels * (2 ** h),
                                                             out_channels=self.hidden_channels * (2 ** (h + 1)),
                                                             head_num=head_num,
                                                             time_dim=self.time_dim,
                                                             num_resnet=num_resnet,
                                                             text_dim=self.text_dim,
                                                             dropout=dropout,
                                                             d_model=d_model,
                                                             dff=dff,
                                                             pos_embed=pos_embed,
                                                             image_size=image_size // (2 ** h)
                                                             ))
        self.encoder = nn.ModuleList(arr)

    def set_decoder(self, num_resnet=4, dropout=.1, head_num=8, d_model=64, dff=512, pos_embed=False, image_size=None):
        arr = []
        for h in range(self.depth, 0, -1):
            arr.append(Up2dBlockConditionalWithTimeAndText(in_channels=self.hidden_channels * (2 ** h),
                                                           out_channels=self.hidden_channels * (2 ** (h - 1)),
                                                           head_num=head_num,
                                                           time_dim=self.time_dim,
                                                           num_resnet=num_resnet,
                                                           text_dim=self.text_dim,
                                                           dropout=dropout,
                                                           d_model=d_model,
                                                           dff=dff,
                                                           pos_embed=pos_embed,
                                                           image_size=image_size // (2 ** (h - 1))
                                                           ))
        self.decoder = nn.ModuleList(arr)

    def set_midlayer(self, dropout=.1, head_num=8, d_model=64, dff=512, pos_embed=False, image_size=32):
        self.mid_layer = nn.ModuleList(
            [
                Resnet2dBlockConditional(in_channels=self.hidden_channels * (2 ** self.depth),
                                         out_channels=self.hidden_channels * (2 ** self.depth),
                                         condition_dim=self.time_dim),
                ImageAndTextCrossAttentionBlock(in_channels=self.hidden_channels * (2 ** self.depth),
                                                context_dim=self.text_dim,
                                                num_head=head_num,
                                                d_model=d_model,
                                                dropout=dropout,
                                                dff=dff,
                                                pos_embed=pos_embed,
                                                image_size=image_size),
                Resnet2dBlockConditional(in_channels=self.hidden_channels * (2 ** self.depth),
                                         out_channels=self.hidden_channels * (2 ** self.depth),
                                         condition_dim=self.time_dim)]
        )

    def forward(self, x, time_embed, text_embed):
        x = self.init(x)
        time_embed_ = self.init_condition(time_embed)
        x = x + time_embed_
        tmp = []
        for index, model in enumerate(self.encoder):
            if not tmp:
                tmp.append(model(x=x, time_embed=time_embed, text_embed=text_embed))
            else:
                tmp.append(model(x=tmp[-1], time_embed=time_embed, text_embed=text_embed))
        mid = self.mid_layer[0](x=tmp.pop(), condition=time_embed)
        mid = self.mid_layer[1](x=mid, text_embed=text_embed)
        mid = self.mid_layer[2](x=mid, condition=time_embed)
        out = None
        for index, model in enumerate(self.decoder):
            if out is None:
                out = model(x=mid, time_embed=time_embed, text_embed=text_embed)
            else:
                t = self.merge_layer[index](tmp.pop(), out)
                out = model(x=t, time_embed=time_embed, text_embed=text_embed)
        out = self.out(out)
        return out
