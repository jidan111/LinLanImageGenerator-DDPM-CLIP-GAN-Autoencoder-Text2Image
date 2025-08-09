from .globals import *


def channels2group_nums(channels):
    """
    用通道数匹配分组数
    :param channels: 通道数
    :return:
    """
    if channels < 8:
        return 0
    arr = [64, 32, 24, 16, 8]
    for i in arr:
        if channels % i == 0:
            return channels // i
    return 1


def scaled_dot_product_attention(query, key, value, attn_mask=None, scale=None):
    """
    :param query: [batch_size, head_num, q_seq_length, d_model]
    :param key: [batch_size, head_num, k_seq_length, d_model]
    :param value: [batch_size, head_num, v_seq_length, d_model]
    :param attn_mask: [batch_size, head_num, q_seq_length, k_seq_length], True为需要填充
    :param scale: 1./sqrt(d_model)
    :return:
    """
    batch_size, head_num, q_seq_length, *_ = query.shape
    batch_size, head_num, k_seq_length, d_model = key.shape
    batch_size, head_num, v_seq_length, *_ = value.shape
    if scale is None:
        scale = 1 / math.sqrt(d_model)
    atten = torch.einsum("bhqd,bhkd->bhqk", query, key) * scale
    if attn_mask is not None:
        atten.masked_fill_(attn_mask, 1e-9)
    score = atten.softmax(-1)
    z = torch.einsum("bhqv,bhvd->bhqd", score, value)
    return z


class SelfAttention(nn.Module):
    """
    自注意力模块,使用torch.nn.functional.scaled_dot_product_attention来实现高效注意力计算
    """

    def __init__(self, query_dim, num_head=8, d_model=64, dropout=.1):
        super(SelfAttention, self).__init__()
        hidden = num_head * d_model
        self.scale = 1. / math.sqrt(d_model)
        self.num_head = num_head
        self.d_model = d_model
        self.Q = nn.Linear(query_dim, hidden, bias=False)
        self.K = nn.Linear(query_dim, hidden, bias=False)
        self.V = nn.Linear(query_dim, hidden, bias=False)
        self.proj_out = nn.Sequential(
            nn.Linear(hidden, query_dim),
            nn.Dropout(dropout)
        )
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.proj_out[0].weight)

    def forward(self, x):
        batch_size, x_seq_length, _ = x.shape
        q = self.Q(x)  # (batch_size, x_seq_length,num_head*hidden_dim)
        k = self.K(x)  # (batch_size, c_seq_length,num_head*hidden_dim)
        v = self.V(x)  # (batch_size, c_seq_length,num_head*hidden_dim)
        q = q.view(batch_size, x_seq_length, self.num_head, self.d_model).transpose(1, 2).contiguous()
        k = k.view(batch_size, x_seq_length, self.num_head, self.d_model).transpose(1, 2).contiguous()
        v = v.view(batch_size, x_seq_length, self.num_head, self.d_model).transpose(1, 2).contiguous()
        z = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None)
        z = z.transpose(1, 2).contiguous().view(batch_size, x_seq_length, -1)
        out = self.proj_out(z)
        return out


class CrossAttention(nn.Module):
    """
    实现图像和文本的特征融合
    """

    def __init__(self, query_dim, context_dim=None, num_head=8, d_model=32, dropout=.1):
        super(CrossAttention, self).__init__()
        hidden = num_head * d_model
        self.have_context = context_dim is not None
        if not self.have_context:
            context_dim = query_dim
        self.scale = 1. / math.sqrt(d_model)
        self.num_head = num_head
        self.d_model = d_model
        self.Q = nn.Linear(query_dim, hidden, bias=False)
        self.K = nn.Linear(context_dim, hidden, bias=False)
        self.V = nn.Linear(context_dim, hidden, bias=False)
        self.proj_out = nn.Sequential(
            nn.Linear(hidden, query_dim),
            nn.Dropout(dropout)
        )
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.proj_out[0].weight)

    def forward(self, x, context=None):
        batch_size, x_seq_length, _ = x.shape
        if not self.have_context:
            context = x
        batch_size, c_seq_length, _ = context.shape
        q = self.Q(x)  # (batch_size, x_seq_length,num_head*hidden_dim)
        k = self.K(context)  # (batch_size, c_seq_length,num_head*hidden_dim)
        v = self.V(context)  # (batch_size, c_seq_length,num_head*hidden_dim)
        q = q.view(batch_size, x_seq_length, self.num_head, self.d_model).transpose(1, 2).contiguous()
        k = k.view(batch_size, c_seq_length, self.num_head, self.d_model).transpose(1, 2).contiguous()
        v = v.view(batch_size, c_seq_length, self.num_head, self.d_model).transpose(1, 2).contiguous()
        z = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None)
        z = z.transpose(1, 2).contiguous().view(batch_size, x_seq_length, -1)
        out = self.proj_out(z)
        return out


class FeedForward(nn.Module):
    def __init__(self, in_dim, dff):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, dff),
            nn.GELU(),
            nn.Linear(dff, in_dim)
        )

    def forward(self, x):
        return self.layer(x)


class ImageAndTextCrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, context_dim, num_head=8, d_model=64, dropout=.1, dff=512, pos_embed=False,
                 image_size=32):
        super(ImageAndTextCrossAttentionBlock, self).__init__()
        self.in_channels = in_channels
        hidden_dim = num_head * d_model
        self.hidden_dim = hidden_dim
        self.have_pos = pos_embed
        group_nums = channels2group_nums(in_channels)
        if group_nums >= 3:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=group_nums)
        else:
            self.norm1 = nn.InstanceNorm2d(num_features=in_channels)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1)
        if self.have_pos is True:
            self.pos_embed = nn.Parameter(torch.randn(size=(image_size * image_size, hidden_dim)))
        self.self_atten = CrossAttention(query_dim=hidden_dim, num_head=num_head, d_model=d_model, dropout=dropout)
        self.cross_atten = CrossAttention(query_dim=hidden_dim, context_dim=context_dim, num_head=num_head,
                                          d_model=d_model, dropout=dropout)
        self.ffc = FeedForward(hidden_dim, dff)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.out_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=in_channels, kernel_size=1)

    def forward(self, x, text_embed=None):
        batch_size, channels, h, w = x.shape
        x_in = x
        x = self.norm1(x)
        x = self.init_conv(x)
        x = x.view(batch_size, self.hidden_dim, -1).transpose(1, 2).contiguous()
        if self.have_pos:
            x = x + self.pos_embed
        x = self.self_atten(self.norm2(x)) + x
        if text_embed is not None:
            x = self.cross_atten(self.norm3(x), context=text_embed) + x
        x = self.ffc(self.norm4(x)) + x
        x = x.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim, h, w)
        x = self.out_conv(x)
        return x + x_in


class ImageSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_head=8, d_model=64, dropout=.1, dff=512, pos_embed=False,
                 image_size=32):
        super(ImageSelfAttentionBlock, self).__init__()
        self.in_channels = in_channels
        hidden_dim = num_head * d_model
        self.hidden_dim = hidden_dim
        self.have_pos = pos_embed
        group_nums = channels2group_nums(in_channels)
        if group_nums >= 3:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=group_nums)
        else:
            self.norm1 = nn.InstanceNorm2d(num_features=in_channels)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1)
        if self.have_pos is True:
            self.pos_embed = nn.Parameter(torch.randn(size=(image_size * image_size, hidden_dim)))
        self.self_atten = SelfAttention(query_dim=hidden_dim, num_head=num_head, d_model=d_model, dropout=dropout)
        self.ffc = FeedForward(hidden_dim, dff)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.out_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x_in = x
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
