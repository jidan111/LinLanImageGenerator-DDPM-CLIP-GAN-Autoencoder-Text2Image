from .globals import *


class CaptionImageDataset(Dataset):
    def __init__(self, item, transform=None):
        self.data = item
        self.transform = transform if transform else self.default_transform()

    def default_transform(self):
        """默认图片预处理流程"""
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为0.5
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),  # 转为 Tensor 并且数值范围变为 [0.0, 1.0]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 数据集的均值
                std=[0.229, 0.224, 0.225]  # ImageNet 数据集的标准差
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['image_id']
        img_path = f"./data/1/image/{image_id:012d}.jpg"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


dataset = ...
dataloader = DataLoader(
    dataset,
    batch_size=176,
    shuffle=True,
    num_workers=8,
    prefetch_factor=4,  # 每个 worker 预取 4 批数据
    pin_memory=True
)
