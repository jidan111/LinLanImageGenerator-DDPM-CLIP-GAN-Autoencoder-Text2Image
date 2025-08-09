from .globals import *


# 获取参数字典
def get_config(**kwargs):
    config = {}
    for k, v in kwargs.items():
        config[k] = v
    return config

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


def count_model_params(model, requires_grad=True):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    return sum(p.numel() for p in model.parameters())


class CaptionImageDataset(Dataset):
    def __init__(self, json_file, transform=None, bpe_dim=128, bpe=None):
        """
        Args:
            json_file (str): caption数据JSON文件路径
            transform (callable): 图片预处理变换
            bpe_dim (int): BPE向量的维度
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.transform = transform if transform else self.default_transform()
        self.bpe_dim = bpe_dim
        self.bpe = bpe  # 假设bpe已在外部初始化

    def default_transform(self):
        """默认图片预处理流程"""
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # 亮度增强，参数0.5表示随机调整亮度在0.5~1.5之间
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为0.5
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),  # 转为 Tensor 并且数值范围变为 [0.0, 1.0]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # ImageNet 数据集的均值
                std=[0.5, 0.5, 0.5]  # ImageNet 数据集的标准差
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图像
        image_id = item['image_id']
        img_path = f"./data/1/image/{image_id:012d}.jpg"
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 编码文本
        caption_vec = self.bpe.encode_sentence(item['caption'], to_tensor=True, dim=self.bpe_dim)
        return image, caption_vec
