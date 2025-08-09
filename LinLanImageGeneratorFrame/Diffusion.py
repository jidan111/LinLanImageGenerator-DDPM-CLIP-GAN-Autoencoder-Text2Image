from .Unet import Unet2dConditionalWithTime, Unet2dConditionalWithTimeAndText
from .globals import *
from .utils import *


class DDPM(nn.Module):
    def __init__(self, model=None, image_shape=None, beta=(1e-4, 0.02), T=100, time_dim=64, schedule_name="sqrt_linear",
                 s=0.008):
        super(DDPM, self).__init__()
        self.config = get_config(model=type(model).__name__ if not hasattr(model, "config") else model.config, image_shape=image_shape, beta=beta, T=T, time_dim=time_dim,
                                 schedule_name=schedule_name, s=s, name=type(self).__name__)
        self.T = T
        self.image_shape = image_shape
        self.time_dim = time_dim
        self.loss = nn.MSELoss()
        self.model = model
        self.time_embd = nn.Embedding(T, time_dim)
        for k, v in self.set_schedule_params(schedule_name=schedule_name, s=s, beta=beta).items():
            self.register_buffer(k, v)

    def get_named_beta_schedule(self, schedule_name, beta=(1e-4, 0.02), s=0.008):
        if schedule_name == "linear":
            scale = 1000 / self.T
            beta_start = scale * beta[0]
            beta_end = scale * beta[1]
            return torch.linspace(
                beta_start, beta_end, self.T
            ).view(-1, 1, 1, 1)
        elif schedule_name == "cosine":
            return self.betas_for_alpha_bar(
                lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2,
                max_beta=0.999
            ).view(-1, 1, 1, 1)
        elif schedule_name == "sqrt_linear":
            sqrt_beta_start = torch.sqrt(torch.tensor(beta[0]))
            sqrt_beta_end = torch.sqrt(torch.tensor(beta[1]))
            return torch.linspace(sqrt_beta_start, sqrt_beta_end, self.T).pow(2).view(-1, 1, 1, 1)
        else:
            raise NotImplementedError(f"没有预设该噪声调度算法: {schedule_name}")

    def betas_for_alpha_bar(self, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(self.T):
            t1 = i / self.T
            t2 = (i + 1) / self.T
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.Tensor(betas)

    def set_schedule_params(self, schedule_name, beta=(1e-4, 0.02), s=0.008):
        beta = self.get_named_beta_schedule(schedule_name=schedule_name, beta=beta, s=s)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        beta_bar = 1 - alpha_bar
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_beta_bar = torch.sqrt(beta_bar)
        sqrt_beta = torch.sqrt(beta)
        sqrt_alpha = torch.sqrt(alpha)
        return {"alpha": alpha, "beta": beta, "sqrt_alpha": sqrt_alpha, "sqrt_beta": sqrt_beta,
                "sqrt_alpha_bar": sqrt_alpha_bar,
                "sqrt_beta_bar": sqrt_beta_bar}

    def add_norise(self, x, index, norise):
        out = self.sqrt_alpha_bar[index] * x + self.sqrt_beta_bar[index] * norise
        return out

    def forward(self, x, text=None):
        norise = torch.randn_like(x)
        index = torch.randint(0, self.T, size=(x.shape[0],)).to(torch.long).to(x.device)
        t = self.time_embd(index)
        target = self.add_norise(x=x, index=index, norise=norise)
        if text is not None:
            pre = self.model(target, t, text)
        else:
            pre = self.model(target, t)
        loss = self.loss(pre, norise)
        return loss

    @torch.no_grad()
    def sample_p(self, x, t, batch_size, text=None):
        index = torch.Tensor([t]).repeat(batch_size).to(torch.long).to(x.device)
        t = self.time_embd(index)
        if text is not None:
            pre = self.model(x, t, text)
        else:
            pre = self.model(x, t)
        norise = torch.randn_like(x).to(x.device)
        mu = (x - (self.beta[index] / self.sqrt_beta_bar[index]) * pre) / self.sqrt_alpha[index]
        x = mu + self.sqrt_beta[index] * norise
        return x

    @torch.no_grad()
    def sample(self, batch_size, x=None, device="cuda", text=None):
        self.model.eval()
        self.time_embd.eval()
        if x is None:
            x = torch.randn(size=(batch_size, *self.image_shape)).to(device)
        batch_size = x.shape[0]
        with torch.no_grad():
            for i in tqdm(range(self.T - 1, -1, -1), desc=f"generate"):
                x = self.sample_p(x=x, t=i, batch_size=batch_size, text=text)
        return x

    @torch.no_grad()
    def ddim_sample_p(self, x, old_t, target_t, batch_size, sigma=0.0, text=None):
        old_index = torch.Tensor([old_t]).repeat(batch_size).to(torch.long).to(x.device)
        target_index = torch.Tensor([target_t]).repeat(batch_size).to(torch.long).to(x.device)
        t_emb = self.time_embd(old_index)
        if text is not None:
            epsilon_theta = self.model(x, t_emb, text)
        else:
            epsilon_theta = self.model(x, t_emb)
        x0_pred = (x - self.sqrt_beta_bar[old_index] * epsilon_theta) / self.sqrt_alpha_bar[old_index]
        sigma = sigma
        noise = torch.randn_like(x)
        x_prev_mean = self.sqrt_alpha_bar[target_index] * x0_pred
        x_prev_var = torch.sqrt(1 - self.sqrt_alpha_bar[target_index] ** 2 - sigma ** 2)
        x_prev_noise = sigma * noise
        x_prev = x_prev_mean + x_prev_var * epsilon_theta + x_prev_noise
        return x_prev

    @torch.no_grad()
    def ddim_sample(self, batch_size, step=2, x=None, sigma=0.0, device="cuda", text=None):
        self.model.eval()
        self.time_embd.eval()
        if x is None:
            x = torch.randn(size=(batch_size, *self.image_shape)).to(device)
        with torch.no_grad():
            for i in tqdm(range(self.T - 1, step, -step), desc="generate"):
                x = self.ddim_sample_p(x=x, old_t=i, target_t=i - step, batch_size=batch_size, sigma=sigma, text=text)
        return x

    @torch.no_grad()
    def save_gif(self, batch_size, text_embed, device="cuda", text=None):
        self.model.eval()
        self.time_embd.eval()
        index = self.T
        row = int(math.sqrt(batch_size))
        x = torch.randn(size=(batch_size, *self.image_shape)).to(device)
        im = make_grid(x, nrow=row, padding=1, normalize=True)
        save_image(im, f"./tmp/{index}.png")
        index -= 1
        for i in tqdm(range(self.T - 1, -1, -1), desc=f"generate"):
            x = self.sample_p(x=x, t=i, text_embed=text_embed, batch_size=batch_size, text=text)
            im = make_grid(x, nrow=row, padding=1, normalize=True)
            save_image(im, f"./tmp/{index}.png")
            index -= 1
