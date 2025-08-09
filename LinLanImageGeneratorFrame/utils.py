from .Losses import *


def set_model_from_config(config, globals_=globals()):
    class_name = config.get("name")
    cls = globals_.get(class_name)
    if not cls or not inspect.isclass(cls):
        raise ValueError(f"未定义该类: {class_name}")
    sig = inspect.signature(cls.__init__)
    valid_params = {}
    for param_name, param in list(sig.parameters.items())[1:]:  # 跳过第一个self参数
        if param_name in config:
            valid_params[param_name] = config[param_name]
        elif param.default is param.empty:  # 检查必需参数
            raise ValueError(f"该参数必须输入: {param_name}")
    return cls(**valid_params)


# 获取参数字典
def get_config(**kwargs):
    config = {}
    for k, v in kwargs.items():
        config[k] = v
    return config


# 计算模型参数量
def count_model_params(model, requires_grad=True):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    return sum(p.numel() for p in model.parameters())


def get_notinstance_params_name(model, pad="&"):
    name = ""
    for key, value in model.__dict__.items():
        if key == "training":
            continue
        if not key.startswith('_') and not isinstance(value, nn.Module):
            name += f"{key}={value}{pad}"
    return name[:-1]


def load_dataloader(path, batch_size=32, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整大小
            transforms.ToTensor(),  # 转为 Tensor 并且数值范围变为 [0.0, 1.0]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 数据集的均值
                std=[0.229, 0.224, 0.225]  # ImageNet 数据集的标准差
            )
        ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=8,
                            prefetch_factor=4,  # 每个 worker 预取 4 批数据
                            pin_memory=True)
    return dataloader


class DiagonalGaussianDistribution(object):
    def __init__(self, tensor, deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__()
        assert tensor.shape[1] % 2 == 0, f"输入的潜在向量无法划分为均值和方差, {tensor.shape[1]}%2 != 0"
        self.params = tensor
        self.mean, self.logvar = tensor.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.deterministic = deterministic
        if deterministic:
            self.var, self.std = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = self.logvar.exp()

    def sample(self):
        out = self.mean + self.std * torch.randn_like(self.mean).to(self.params.device)
        return out

    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1. - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum((self.mean - other.mean).pow(
                    2) / other.var + self.var / other.var - 1. - self.logvar + other.logvar, dim=[1, 2, 3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logwopi = np.log(2. * np.pi)
        return 0.5 * torch.sum(logwopi + self.logvar + (sample - self.mean).pow(2) / self.var, dim=[1, 2, 3])


def get_load_state_dict_from_compile(file, device="cuda"):
    new_dict = OrderedDict()
    for k, v in torch.load(file, map_location=device).items():
        key = k.replace("_orig_mod.", "")
        new_dict[key] = v
    return new_dict
