from .globals import *


def NLL_Q2Unknown(q_mean, q_logvar, unknown_sample):
    logwopi = np.log(2. * np.pi)
    nll = 0.5 * (logwopi + q_logvar + (unknown_sample - q_mean).pow(2) * torch.exp(-q_logvar))
    return nll.sum(dim=[1, 2, 3])


def KL_Q2P(q_mean, q_logvar, p_mean, p_logvar):
    kl = 0.5 * (-1 + p_logvar - q_logvar + torch.exp(q_logvar - p_logvar) + (q_mean - p_mean).pow(2) * torch.exp(
        -p_logvar))
    return kl.sum(dim=[1, 2, 3])


def KL_Q2Normal(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])


def standard_normal_cdf(x):
    return 0.5 * (1.0 + F.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x.pow(3))))


def compute_gradient_penalty(D, real_samples, fake_samples, create_graph=True, retain_graph=True):
    alpha = torch.rand(size=(real_samples.shape[0], 1, 1, 1)).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(real_samples.device)
    d_interpolates = D(interpolates)
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
    return gradient_penalty


def calculate_adaptive_weight(a_loss, b_loss, model_last_layer):
    """
    :param a_loss: 主损失
    :param b_loss: 次损失
    :param model_last_layer: 模型最后一层
    :return:
    """
    a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
    b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
    b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
    b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
    return b_weight


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def hinge_discriminator_loss(pre, y):
    loss_y = torch.mean(F.relu(1. - y))
    loss_pre = torch.mean(F.relu(1. + pre))
    d_loss = loss_y + loss_pre
    return d_loss
