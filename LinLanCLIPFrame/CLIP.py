from .struct import *
from .ImageFeatures import *
from .utils import *


class ImageEncoderVIT(nn.Module):
    def __init__(self, image_shape, patch_size, embed_dim=64, encoder_layer_num=6, head_num=8,
                 d_model=64, dropout=.1, dff=512, out_dim=512):
        super(ImageEncoderVIT, self).__init__()
        self.image_shape = image_shape
        self.config = get_config(image_shape=image_shape, patch_size=patch_size, embed_dim=embed_dim,
                                 encoder_layer_num=encoder_layer_num, head_num=head_num, d_model=d_model,
                                 dropout=dropout, dff=dff, out_dim=out_dim)
        if type(patch_size) in (tuple, list):
            assert image_shape[1] % patch_size[0] == 0 and image_shape[2] % patch_size[1] == 0, "图片大小无法被patch大小整除"
            self.seq_length = (image_shape[1] // patch_size[0]) * (image_shape[2] // patch_size[1])
            self.h = image_shape[1] // patch_size[0]
            self.w = image_shape[2] // patch_size[1]
        else:
            assert image_shape[1] % patch_size == 0, "图片大小无法被patch大小整除"
            self.seq_length = (image_shape[1] // patch_size) ** 2
            self.h = self.w = image_shape[1] // patch_size
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.init_conv = nn.Conv2d(in_channels=image_shape[0], out_channels=embed_dim, kernel_size=patch_size,
                                   stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(size=(self.seq_length + 1, self.embed_dim)))
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.ModuleList(
            [EncoderLayer(query_dim=embed_dim, d_model=d_model, head_num=head_num, dropout=dropout, dff=dff)
             for i in range(encoder_layer_num)])
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Parameter(torch.randn(size=(embed_dim, out_dim)))

    def encode(self, q, q_mask=None):
        for layer in self.encoder:
            q = layer(q, q_mask)
        return q

    def forward(self, x):
        batch_size, *_ = x.shape
        x = self.init_conv(x)
        x = x.view(batch_size, self.embed_dim, self.seq_length).transpose(1,
                                                                          2).contiguous()  # (batch_size, seq_length, embed_dim)
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.norm1(x)
        encode = self.encode(q=x, q_mask=None)
        encode = self.norm2(encode[:, 0, :])
        out = torch.matmul(encode, self.proj_out)
        return out

    @torch.no_grad()
    def encode_image(self, image):
        out = self(image)
        out = out / out.norm(dim=1, keepdim=True)
        return out


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, max_seq_length=512, d_model=64, head_num=8, encoder_layer_num=6,
                 dff=512, dropout=.1, out_dim=512):
        super(TextEncoder, self).__init__()
        self.config = get_config(vocab_size=vocab_size, embed_dim=embed_dim, max_seq_length=max_seq_length,
                                 d_model=d_model, head_num=head_num, encoder_layer_num=encoder_layer_num, dff=dff,
                                 dropout=dropout, out_dim=out_dim)
        self.vocab_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_seq_length + 1, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)))
        self.encoder = nn.ModuleList(
            [EncoderLayer(query_dim=embed_dim, d_model=d_model, head_num=head_num, dff=dff, dropout=dropout) for i in
             range(encoder_layer_num)])
        self.proj_out = nn.Parameter(torch.randn(size=(embed_dim, out_dim)))

    def make_q_mask(self, x):
        mask = torch.ones(size=(x.shape[0], x.shape[1])).to(device=x.device)
        mask.masked_fill_(x == 0, 0)
        mask = torch.cat([
            torch.ones((x.shape[0], 1), device=x.device),
            mask
        ], dim=1).bool()
        return mask.unsqueeze(1).unsqueeze(1)  # True表示有效，False表示无效

    def encode(self, q, mask=None):
        for layer in self.encoder:
            q = layer(q=q, mask=mask)
        return q

    def forward(self, text):
        q_mask = self.make_q_mask(text)
        pos_indices = torch.arange(text.shape[1] + 1, device=text.device).unsqueeze(0)
        position = self.position_embed(pos_indices)
        x = self.vocab_embed(text)
        cls_token = self.cls_token.expand(text.shape[0], 1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + position
        x = self.norm(x)
        encode_out = self.encode(x, mask=q_mask)
        out = encode_out[:, 0, :]
        out = torch.matmul(out, self.proj_out)
        return out

    @torch.no_grad()
    def encode_text(self, text):
        out = self(text)
        out = out / out.norm(dim=1, keepdim=True)
        return out


class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self):
        return self.scale.clamp(min=-10, max=10)


class CLIP(nn.Module):
    def __init__(self, image_encoder_type="conv", image_shape=None, patch_size=16, vocab_size=100, embed_dim=512,
                 max_seq_length=128, out_dim=512,
                 head_num=8, encoder_layer_num=6, d_model=64, dropout=.1, dff=512, min_depth=2, hidden_channels=8,
                 min_size=8, kernel_size=7):
        super(CLIP, self).__init__()
        assert image_encoder_type in ["conv", "vit"], "只支持conv和vit两种类型的图片提取器"
        self.config = get_config(image_encoder_type=image_encoder_type, image_shape=image_shape, patch_size=patch_size,
                                 vocab_size=vocab_size, embed_dim=embed_dim, max_seq_length=max_seq_length,
                                 out_dim=out_dim, head_num=head_num, encoder_layer_num=encoder_layer_num,
                                 d_model=d_model, dropout=dropout, dff=dff, min_depth=min_depth,
                                 hidden_channels=hidden_channels, min_size=min_size, kernel_size=kernel_size)
        if image_encoder_type == "conv":
            self.image_encoder = Conv2d_Features(image_shape=image_shape, min_depth=min_depth,
                                                 hidden_channels=hidden_channels, min_size=min_size,
                                                 kernel_size=kernel_size)
        else:
            self.image_encoder = ImageEncoderVIT(image_shape=image_shape, patch_size=patch_size, embed_dim=embed_dim,
                                                 out_dim=out_dim, encoder_layer_num=encoder_layer_num, d_model=d_model,
                                                 head_num=head_num, dropout=dropout, dff=dff)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim, out_dim=out_dim,
                                        max_seq_length=max_seq_length, d_model=d_model, head_num=head_num,
                                        encoder_layer_num=encoder_layer_num, dff=dff, dropout=dropout)
        self.logit_scale = Scale()

    def forward(self, image, text):
        assert image.shape[0] == text.shape[0], "batch维度不一致"
        image_encode = self.image_encoder(image)
        text_encode = self.text_encoder(text)
        image_encode = image_encode / image_encode.norm(dim=1, keepdim=True)
        text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
        log_scale = self.logit_scale().exp()
        image_pre = log_scale * torch.matmul(image_encode, text_encode.t())
        text_pre = image_pre.t()
        labels = torch.arange(len(image_pre), device=text_pre.device)
        loss = (F.cross_entropy(image_pre, labels) + F.cross_entropy(text_pre, labels)
                ) / 2
        return loss

    @torch.no_grad()
    def encode_image(self, x):
        out = self.image_encoder(x)
        out = out / out.norm(dim=1, keepdim=True)
        return out

    @torch.no_grad()
    def encode_text(self, x):
        out = self.text_encoder(x)
        out = out / out.norm(dim=1, keepdim=True)
        return out

    @torch.no_grad()
    def accuracy(self, image, text):
        assert image.shape[0] == text.shape[0], "batch维度不一致"
        batch_size, *_ = image.shape
        image_encode = self.image_encoder(image)
        text_encode = self.text_encoder(text)
        image_encode = image_encode / image_encode.norm(dim=1, keepdim=True)
        text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
        log_scale = self.logit_scale().exp()
        image_pre = log_scale * torch.matmul(image_encode, text_encode.t())
        text_pre = image_pre.t()
        image_index = image_pre.argmax(-1)
        text_index = text_pre.argmax(-1)
        labels = torch.arange(text_pre.shape[0]).to(text_pre.device)
        image_cnt = torch.sum(image_index == labels)
        text_cnt = torch.sum(text_index == labels)
        print("image2text准确率:{:.2%}, text2image准确率:{:.2%}".format(image_cnt / batch_size, text_cnt / batch_size))
