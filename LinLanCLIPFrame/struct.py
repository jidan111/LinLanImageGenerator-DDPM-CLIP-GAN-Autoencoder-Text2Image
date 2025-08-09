from .globals import *


def default(target, other):
    if target is not None:
        return target
    return other


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim=None, value_dim=None, head_num=8, d_model=64, dropout=.1):
        super(MultiheadAttentionBlock, self).__init__()
        key_dim = default(other=query_dim, target=key_dim)
        value_dim = default(other=query_dim, target=value_dim)
        hidden_dim = head_num * d_model
        self.head_num = head_num
        self.d_model = d_model
        self.scale = 1. / math.sqrt(d_model)
        self.Q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.K = nn.Linear(key_dim, hidden_dim, bias=False)
        self.V = nn.Linear(value_dim, hidden_dim, bias=False)
        self.projout = nn.Sequential(
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v, mask):
        batch_size, q_seq_length, *_ = q.shape
        batch_size, k_seq_length, *_ = k.shape
        batch_size, v_seq_length, *_ = v.shape
        if mask is not None:
            assert mask.shape[-1] == k_seq_length, "掩码维度需要与键维度一致"
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q = q.view(batch_size, q_seq_length, self.head_num, self.d_model).transpose(1, 2).contiguous()
        k = k.view(batch_size, k_seq_length, self.head_num, self.d_model).transpose(1, 2).contiguous()
        v = v.view(batch_size, v_seq_length, self.head_num, self.d_model).transpose(1, 2).contiguous()
        z = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, scale=self.scale)
        z = z.transpose(1, 2).contiguous().view(batch_size, q_seq_length, -1)
        out = self.projout(z)
        return out
    # def forward(self, q, k, v, mask):
    #     batch_size, q_seq_length, *_ = q.shape
    #     batch_size, k_seq_length, *_ = k.shape
    #     batch_size, v_seq_length, *_ = v.shape
    #     q = self.Q(q)
    #     k = self.K(k)
    #     v = self.V(v)
    #     q = q.view(batch_size, q_seq_length, self.head_num, self.d_model)
    #     k = k.view(batch_size, k_seq_length, self.head_num, self.d_model)
    #     v = v.view(batch_size, v_seq_length, self.head_num, self.d_model)
    #     atten = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale
    #     if mask is not None:
    #         assert mask.shape[1] == k_seq_length, "掩码维度需要与键维度一致"
    #         # mask shape: (batch_size, k_seq_length)
    #         mask = mask.unsqueeze(1).unsqueeze(1)
    #         atten = atten.masked_fill(~mask.bool(), torch.finfo(atten.dtype).min)
    #     score = atten.softmax(dim=-1)
    #     z = torch.einsum("bhqk,bkhd->bqhd", score, v).view(batch_size, q_seq_length, -1)
    #     out = self.projout(z)
    #     return out


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.layer(x)


class EncoderLayer(nn.Module):
    def __init__(self, query_dim, key_dim=None, value_dim=None, d_model=64, head_num=8, dff=512, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.atten = MultiheadAttentionBlock(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim,
                                             head_num=head_num, d_model=d_model, dropout=dropout)
        self.ffc = FeedForward(in_dim=query_dim, hidden_dim=dff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, mask=None):
        residual = q
        q = self.norm1(q)
        atten = self.atten(q=q, k=q, v=q, mask=mask)
        q = residual + self.dropout1(atten)
        residual = q
        q = self.norm2(q)
        out = self.ffc(q)
        out = residual + self.dropout2(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, query_dim, key_dim=None, value_dim=None, d_model=64, head_num=8, dff=512, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.self_atten = MultiheadAttentionBlock(query_dim=query_dim, key_dim=query_dim, value_dim=query_dim,
                                                  head_num=head_num, d_model=d_model, dropout=dropout)
        key_dim = default(target=key_dim, other=query_dim)
        value_dim = default(target=value_dim, other=key_dim)
        self.cross_atten = MultiheadAttentionBlock(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim,
                                                   head_num=head_num, d_model=d_model, dropout=dropout)
        self.ffc = FeedForward(in_dim=query_dim, hidden_dim=dff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, q=None, k=None, q_mask=None, k_mask=None):
        residual = q
        q = self.norm1(q)
        self_atten = self.self_atten(q=q, k=q, v=q, mask=q_mask)
        q = residual + self.dropout1(self_atten)
        residual = q
        q = self.norm2(q)
        cross_atten = self.cross_atten(q=q, k=k, v=k, mask=k_mask)
        atten = residual + self.dropout2(cross_atten)
        residual = atten
        atten = self.norm3(atten)
        ffc = self.ffc(atten)
        out = residual + self.dropout3(ffc)
        return out
