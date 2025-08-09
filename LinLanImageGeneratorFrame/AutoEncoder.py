from .struct import *
from .utils import *
from .Discriminator import DiscriminatorModel


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=1):
        super(VectorQuantizer, self).__init__()
        self.config = get_config(n_e=n_e, e_dim=e_dim, beta=beta, name=type(self).__name__)
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', self.embedding.weight))
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
               torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = torch.einsum('b h w c -> b c h w', z_q).contiguous()
        return z_q, loss, encoding_indices

    @torch.no_grad()
    def get_codebook_entry(self, indices, shape):
        # 图片形状(batch,channels,h,w)，一般做离散化的时候才需要
        shape = (shape[0], shape[2], shape[3], shape[1])
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class LatentVAE2d(nn.Module):
    def __init__(self, image_shape: list,
                 depth: init,
                 hidden_channels: init,
                 latent_dim: init,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False):
        super(LatentVAE2d, self).__init__()
        assert image_shape[1] % 2 ** depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{2 ** depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.config = get_config(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels,
                                 latent_dim=latent_dim, attention=attention, head_num=head_num, d_model=d_model,
                                 num_resnet=num_resnet, dropout=dropout, dff=dff, pos_embed=pos_embed,
                                 name=type(self).__name__)
        self.image_shape = image_shape
        self.depth = depth
        self.latent_dim = latent_dim
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
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.q_enc = nn.Conv2d(in_channels=latent_dim * 2,
                               out_channels=latent_dim * 2, kernel_size=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=self.hidden_channels * (2 ** self.depth),
                               kernel_size=1)

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
            if h == self.depth - 1:
                arr.append(Down2dBlock(in_channels=self.hidden_channels * (2 ** h),
                                       out_channels=2 * self.latent_dim,
                                       attention=attention[index],
                                       head_num=head_num,
                                       d_model=d_model,
                                       num_resnet=num_resnet,
                                       dropout=dropout,
                                       dff=dff,
                                       pos_embed=pos_embed,
                                       image_size=in_size // (2 ** index)))
            else:
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
        self.encoder = nn.Sequential(*arr)

    def set_decoder(self, attention=False,
                    head_num=8,
                    d_model=64,
                    num_resnet=4,
                    dropout=.1,
                    dff=512,
                    pos_embed=False,
                    in_size=32):
        if not attention:
            attention = [False] * self.depth
        attention = attention[::-1]
        arr = []
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
        self.decoder = nn.Sequential(*arr)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.image_shape[0], kernel_size=1),
            nn.Tanh()
        )

    def get_decoder_last_layer(self):
        return self.out[0].weight

    def get_encode_last_layer(self):
        return self.q_enc.weight

    def encode(self, x):
        h = self.encoder(x)
        q_enc = self.q_enc(h)
        q_prior = DiagonalGaussianDistribution(q_enc)
        return q_prior

    def decode(self, x):
        x = self.p_dec(x)
        out = self.decoder(x)
        out = self.out(out)
        return out

    def forward(self, x, return_z=False):
        x = self.init(x)
        q_prior = self.encode(x)
        z = q_prior.sample()
        out = self.decode(z)
        if return_z:
            return out, z
        return out, q_prior

    @torch.no_grad()
    def latent_space(self, x):
        with torch.no_grad():
            x = self.init(x)
            h = self.encoder(x)
            q_enc = self.q_enc(h)
        return q_enc

    @torch.no_grad()
    def encode_image(self, x):
        x = self.init(x)
        q_prior = self.encode(x)
        return q_prior.sample()

    @torch.no_grad()
    def decode_latent(self, x):
        out = self.decode(x)
        return out


class LatentVQ_VAE2d(nn.Module):
    def __init__(self, image_shape: tuple,
                 depth: init,
                 hidden_channels: init,
                 latent_dim: init,
                 attention=False,
                 head_num=8,
                 d_model=64,
                 num_resnet=4,
                 dropout=.1,
                 dff=512,
                 pos_embed=False,
                 n_embed=64,
                 beta=0.25):
        super(LatentVQ_VAE2d, self).__init__()
        assert image_shape[1] % 2 ** depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{2 ** depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.config = get_config(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels,
                                 latent_dim=latent_dim, attention=attention, head_num=head_num, d_model=d_model,
                                 num_resnet=num_resnet, dropout=dropout, dff=dff, pos_embed=pos_embed, n_embed=n_embed,
                                 beta=beta, name=type(self).__name__)
        self.image_shape = image_shape
        self.depth = depth
        self.latent_dim = latent_dim
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
        self.dim = image_shape[1] // (2 ** depth)
        self.VectorQuantizer = VectorQuantizer(n_e=n_embed, e_dim=latent_dim, beta=beta)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.q_enc = nn.Conv2d(in_channels=self.hidden_channels * (2 ** self.depth),
                               out_channels=latent_dim, kernel_size=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=self.hidden_channels * (2 ** self.depth),
                               kernel_size=1)

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
        self.encoder = nn.Sequential(*arr)

    def set_decoder(self, attention=False,
                    head_num=8,
                    d_model=64,
                    num_resnet=4,
                    dropout=.1,
                    dff=512,
                    pos_embed=False,
                    in_size=32):
        if not attention:
            attention = [False] * self.depth
        attention = attention[::-1]
        arr = []
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
        self.decoder = nn.Sequential(*arr)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.image_shape[0], kernel_size=1),
            nn.Tanh()
        )

    def get_decoder_last_layer(self):
        return self.out[0].weight

    def encode(self, x):
        h = self.encoder(x)
        q_enc = self.q_enc(h)
        q_prior, book_loss, index = self.VectorQuantizer(q_enc)
        return q_prior, book_loss, index

    def decode(self, x):
        x = self.p_dec(x)
        out = self.decoder(x)
        out = self.out(out)
        return out

    def forward(self, x):
        x = self.init(x)
        z, book_loss, index = self.encode(x)
        out = self.decode(z)
        return out, book_loss

    @torch.no_grad()
    def encode_image(self, x):
        x = self.init(x)
        q_prior, book_loss, index = self.encode(x)
        return q_prior

    @torch.no_grad()
    def decode_latent(self, x):
        out = self.decode(x)
        return out


def l1(pre, y):
    return torch.abs(y - pre)


def l2(pre, y):
    return torch.square(y - pre)


def hinge_discriminator_loss(pre, y):
    loss_y = torch.mean(F.relu(1. - y))
    loss_pre = torch.mean(F.relu(1. + pre))
    d_loss = 0.5 * (loss_y + loss_pre)
    return d_loss


class Logvar(nn.Module):
    def __init__(self):
        super(Logvar, self).__init__()
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)

    def forward(self):
        return torch.clamp(self.logvar, min=-10, max=10)


class LossWithDiscriminator(nn.Module):
    def __init__(self, image_shape, depth, hidden_channels, net="alex", rec_weight=1., lpips_weight=.5, kl_weight=0.3,
                 discriminator_type="hinge", have_lpips=True, rec_type="l1", reduction="sum"):
        super(LossWithDiscriminator, self).__init__()
        assert discriminator_type in ["hinge", "w-gp"], "只支持hinge和W-GP判别器损失"
        assert rec_type in ["l1", "l2"], "只支持l1和l2两种重构损失"
        self.config = get_config(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels, net=net,
                                 rec_weight=rec_weight, lpips_weight=lpips_weight, kl_weight=kl_weight,
                                 discriminator_type=discriminator_type, have_lpips=have_lpips, rec_type=rec_type,
                                 reduction=reduction, name=type(self).__name__)
        self.discriminator_type = discriminator_type
        self.have_lpips = have_lpips
        self.discriminator = DiscriminatorModel(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels)
        if discriminator_type == "hinge":
            self.d_loss = hinge_discriminator_loss
        else:
            self.d_loss = WGAN_GP_DLoss()
        self.rec_type = rec_type
        if rec_type == "l1":
            self.rec_loss = l1
        else:
            self.rec_loss = l2
        self.rec_weight = rec_weight
        self.logvar = Logvar()
        if self.have_lpips:
            self.p_loss = LPIPS(net=net).eval()
        self.lpips_weight = lpips_weight
        self.reduction = reduction
        if self.reduction == "mean":
            self.weight = (image_shape[0] * image_shape[1] * image_shape[2])
        self.kl_weight = kl_weight

    def calculate_adaptive_weight(self, a_loss, b_loss, model_last_layer):
        a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
        b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
        return b_weight

    def forward(self, vae_input, vae_out, latent=None, vae_last_layer_weight=None, mode="vae", use_gan=True):
        assert mode in ["vae", "dis"], "只支持vae和dis两种训练"
        if mode == "vae":
            batch_size = vae_out.shape[0]
            logvar = self.logvar()
            rec_loss = self.rec_weight * self.rec_loss(vae_input, vae_out)
            if self.have_lpips:
                rec_loss = rec_loss + self.lpips_weight * self.p_loss(vae_input, vae_out)
            nll_loss = rec_loss / torch.exp(logvar + 1e-6) + logvar
            if self.reduction == "mean":
                kl_loss = latent.kl().sum() / (batch_size * self.weight)
                nll_loss = nll_loss.sum() / (batch_size * self.weight)
            else:
                kl_loss = self.kl_weight * latent.kl().sum() / batch_size
                nll_loss = nll_loss.sum() / batch_size
            vae_loss = nll_loss + kl_loss
            if use_gan is False:
                return vae_loss
            self.discriminator.eval()
            g_loss = -self.discriminator(vae_out).sum() / batch_size
            g_weight = self.calculate_adaptive_weight(vae_loss, g_loss, vae_last_layer_weight)
            loss = vae_loss + g_weight * g_loss
            return loss
        else:
            self.discriminator.train()
            if self.discriminator_type == "hinge":
                dis_false = self.discriminator(vae_out)
                dis_true = self.discriminator(vae_input)
                d_loss = self.d_loss(dis_false, dis_true)
            else:
                d_loss = self.d_loss(model=self.discriminator, real_samples=vae_input, fake_samples=vae_out,
                                     create_graph=True, retain_graph=True)
            return d_loss


class AutoEncoderKL(nn.Module):
    def __init__(self, image_shape, vae_hidden_channels, vae_depth, vae_latent_dim, vae_attention, dis_depth,
                 dis_hidden_channels, net="alex", rec_weight=1., lpips_weight=.5, kl_weight=0.3,
                 discriminator_type="hinge", have_lpips=True, rec_type="l1", reduction="mean", num_resnet=4):
        super(AutoEncoderKL, self).__init__()
        self.config = get_config(image_shape=image_shape, vae_hidden_channels=vae_hidden_channels, vae_depth=vae_depth,
                                 vae_latent_dim=vae_latent_dim, vae_attention=vae_attention, dis_depth=dis_depth,
                                 dis_hidden_channels=dis_hidden_channels, net=net, rec_weight=rec_weight,
                                 lpips_weight=lpips_weight, kl_weight=kl_weight, discriminator_type=discriminator_type,
                                 have_lpips=have_lpips, rec_type=rec_type, reduction=reduction, num_resnet=num_resnet,
                                 name=type(self).__name__)
        self.vae = LatentVAE2d(image_shape=image_shape, hidden_channels=vae_hidden_channels, depth=vae_depth,
                               attention=vae_attention, latent_dim=vae_latent_dim, num_resnet=num_resnet)
        self.loss = LossWithDiscriminator(image_shape=image_shape, hidden_channels=dis_hidden_channels,
                                          depth=dis_depth, net=net, lpips_weight=lpips_weight, kl_weight=kl_weight,
                                          discriminator_type=discriminator_type, rec_weight=rec_weight,
                                          have_lpips=have_lpips, rec_type=rec_type, reduction=reduction)

    def set_optim(self, lr1=1e-4, lr2=5e-4):
        vae_optim = torch.optim.Adam(params=list(self.vae.parameters()) + list(self.loss.logvar.parameters()),
                                     lr=lr1,
                                     betas=(0.5, 0.9))
        dis_optim = torch.optim.Adam(params=self.loss.discriminator.parameters(), lr=lr2, betas=(0.5, 0.9))
        return vae_optim, dis_optim

    def forward(self, x):
        out, latent = self.vae(x)
        return out, latent

    def training_step(self, x, mode="vae", use_gan=False):
        assert mode in ["vae", "dis"], "只支持训练vae和dis"
        out, latent = self(x)
        if mode == "vae":
            loss = self.loss(vae_input=x, vae_out=out, latent=latent,
                             vae_last_layer_weight=self.vae.get_decoder_last_layer(), mode=mode, use_gan=use_gan)
            return loss
        else:
            loss = self.loss(vae_input=x, vae_out=out.detach(), mode=mode)
            return loss


class VQLossWithDiscriminator(nn.Module):
    def __init__(self, image_shape, depth, hidden_channels, net="alex", rec_weight=1., lpips_weight=.5, book_weight=0.3,
                 discriminator_type="hinge", have_lpips=True, rec_type="l1"):
        super(VQLossWithDiscriminator, self).__init__()
        assert rec_type in ["l1", l2], "只支持l1和l2两种重构损失"
        assert discriminator_type in ["hinge", "w-gp"], "只支持hinge和W-GP判别器损失"
        self.discriminator_type = discriminator_type
        self.have_lpips = have_lpips
        self.discriminator = DiscriminatorModel(image_shape=image_shape, depth=depth, hidden_channels=hidden_channels)
        if discriminator_type == "hinge":
            self.d_loss = hinge_discriminator_loss
        else:
            self.d_loss = WGAN_GP_DLoss()
        self.rec_type = rec_type
        if rec_type == "l1":
            self.rec_loss = l1
        else:
            self.rec_loss = l2
        self.rec_weight = rec_weight
        self.logvar = Logvar()
        if self.have_lpips:
            self.p_loss = LPIPS(net=net).eval()
        self.lpips_weight = lpips_weight
        self.book_weight = book_weight

    def calculate_adaptive_weight(self, a_loss, b_loss, model_last_layer):
        a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
        b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
        return b_weight

    def forward(self, vae_input, vae_out, book_loss=None, vae_last_layer_weight=None, mode="vae", use_gan=True):
        assert mode in ["vae", "dis"], "只支持vae和dis两种训练"
        if mode == "vae":
            logvar = self.logvar()
            rec_loss = self.rec_weight * self.rec_loss(vae_input, vae_out)
            if self.have_lpips:
                rec_loss = rec_loss + self.lpips_weight * self.p_loss(vae_input, vae_out)
            nll_loss = rec_loss / torch.exp(logvar + 1e-6) + logvar
            nll_loss = nll_loss.mean()
            vae_loss = nll_loss + self.book_weight * book_loss
            if use_gan is False:
                return vae_loss
            self.discriminator.eval()
            g_loss = -self.discriminator(vae_out).mean()
            g_weight = self.calculate_adaptive_weight(vae_loss, g_loss, vae_last_layer_weight)
            loss = vae_loss + g_weight * g_loss
            return loss
        else:
            self.discriminator.train()
            if self.discriminator_type == "hinge":
                dis_false = self.discriminator(vae_out)
                dis_true = self.discriminator(vae_input)
                d_loss = self.d_loss(dis_false, dis_true)
            else:
                d_loss = self.d_loss(model=self.discriminator, real_samples=vae_input, fake_samples=vae_out,
                                     create_graph=True, retain_graph=True)
            return d_loss


class VQAutoEncoderKL(nn.Module):
    def __init__(self, image_shape, vae_hidden_channels, vae_depth, vae_latent_dim, vae_attention, vae_n_embed,
                 dis_depth, vae_beta=0.25, book_weight=1.,
                 dis_hidden_channels=8, net="alex", rec_weight=1., lpips_weight=.5,
                 discriminator_type="hinge", have_lpips=True, rec_type="l1", num_resnet=4):
        super(VQAutoEncoderKL, self).__init__()
        self.config = get_config(image_shape=image_shape, vae_hidden_channels=vae_depth, vae_latent_dim=vae_latent_dim,
                                 vae_attention=vae_attention, vae_n_embed=vae_n_embed, dis_depth=dis_depth,
                                 vae_beta=vae_beta, book_weight=book_weight, dis_hidden_channels=dis_hidden_channels,
                                 net=net, rec_weight=rec_weight, lpips_weight=lpips_weight,
                                 discriminator_type=discriminator_type, have_lpips=have_lpips, rec_type=rec_type,
                                 num_resnet=num_resnet, name=type(self).__name__)
        self.vae = LatentVQ_VAE2d(image_shape=image_shape, hidden_channels=vae_hidden_channels, depth=vae_depth,
                                  attention=vae_attention, latent_dim=vae_latent_dim, n_embed=vae_n_embed,
                                  beta=vae_beta, num_resnet=num_resnet)
        self.loss = VQLossWithDiscriminator(image_shape=image_shape, hidden_channels=dis_hidden_channels,
                                            depth=dis_depth, net=net, lpips_weight=lpips_weight,
                                            book_weight=book_weight,
                                            discriminator_type=discriminator_type, rec_weight=rec_weight,
                                            have_lpips=have_lpips, rec_type=rec_type)

    def set_optim(self, lr1=1e-4, lr2=5e-4):
        vae_optim = torch.optim.Adam(params=list(self.vae.parameters()) + list(self.loss.logvar.parameters()),
                                     lr=lr1,
                                     betas=(0.5, 0.9))
        dis_optim = torch.optim.Adam(params=self.loss.discriminator.parameters(), lr=lr2, betas=(0.5, 0.9))
        return vae_optim, dis_optim

    def forward(self, x):
        out, book_loss = self.vae(x)
        return out, book_loss

    def training_step(self, x, mode="vae", use_gan=False):
        assert mode in ["vae", "dis"], "只支持训练vae和dis"
        out, book_loss = self(x)
        if mode == "vae":
            loss = self.loss(vae_input=x, vae_out=out, book_loss=book_loss,
                             vae_last_layer_weight=self.vae.get_decoder_last_layer(), mode=mode, use_gan=use_gan)
            return loss
        else:
            loss = self.loss(vae_input=x, vae_out=out.detach(), mode=mode, use_gan=True)
            return loss
