from .functions import *

class VAELoss(nn.Module):
    def __init__(self, rec_weight=1., kl_weight=.3):
        super(VAELoss, self).__init__()
        self.rec_weight = rec_weight
        self.kl_weight = kl_weight

    def KL(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])

    def forward(self, pre, y, mean=None, logvar=None, latent=None):
        batch_size = pre.shape[0]
        rec_loss = torch.abs(pre - y).sum() / batch_size
        if latent is not None:
            kl_loss = latent.kl().mean()
        else:
            kl_loss = self.KL(mean, logvar).mean()
        loss = self.rec_weight * rec_loss + self.kl_weight * kl_loss
        return loss


class VQVAELoss(nn.Module):
    def __init__(self, book_loss_weight=1.):
        super(VQVAELoss, self).__init__()
        self.book_loss_weight = book_loss_weight

    def forward(self, pre, y, book_loss):
        rec_loss = torch.abs(y - pre).mean()
        return rec_loss + self.book_loss_weight * book_loss


class WGAN_GP_DLoss(nn.Module):
    def __init__(self, lambda_gp=10):
        super(WGAN_GP_DLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, model, real_samples, fake_samples, create_graph=True, retain_graph=True):
        alpha = torch.rand(size=(real_samples.shape[0], 1, 1, 1)).to(real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(
            real_samples.device)
        d_interpolates = model(interpolates)
        fake = torch.ones(size=(real_samples.shape[0], 1), requires_grad=False).to(real_samples.device)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        true_score = model(real_samples)
        fake_score = model(fake_samples)
        return -true_score.mean() + fake_score.mean() + self.lambda_gp * gradient_penalty


class WGAN_Hinge_DLoss(nn.Module):
    def __init__(self):
        super(WGAN_Hinge_DLoss, self).__init__()

    def forward(self, pre, y):
        return hinge_discriminator_loss(pre=pre, y=y)
