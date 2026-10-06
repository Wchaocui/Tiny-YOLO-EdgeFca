import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import lpips
from PIL import Image
from FourLLIE.models.archs import FourLLIE
import os
from torchvision.transforms import Compose, ToTensor, CenterCrop, RandomCrop

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, crop_size=320, random_crop=False):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loaded {len(self.image_files)} images from {root_dir}")  # 调试信息

        # 如果指定了裁剪尺寸，构建裁剪变换
        if self.crop_size:
            if self.random_crop:
                self.crop_transform = RandomCrop(self.crop_size)
            else:
                self.crop_transform = CenterCrop(self.crop_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # 如果指定了裁剪变换，对图像进行裁剪
        if self.crop_size:
            image = self.crop_transform(image)

        # 应用其他变换
        if self.transform:
            image = self.transform(image)
        return image

# 定义无监督损失函数
class UnsupervisedLoss(nn.Module):
    def __init__(self):
        super(UnsupervisedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = lpips.LPIPS(net='alex').to(device)

    def forward(self, input_image, enhanced_image):
        # 亮度一致性损失
        brightness_loss = self.l1_loss(enhanced_image.mean(dim=[1, 2, 3]), input_image.mean(dim=[1, 2, 3]))

        # 感知损失
        perceptual_loss = self.perceptual_loss(enhanced_image, input_image).mean()

        # 总损失
        total_loss = brightness_loss + perceptual_loss
        return total_loss

# 加载数据集
dataset = ImageDataset(root_dir='/home/s0433/local/data/spark-2022-stream-1/images/test', transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型和优化器
model = FourLLIE.FourLLIE(nf=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = UnsupervisedLoss()
num_epochs = 25
# 训练模型
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()
        enhanced_image, _, _, _ = model(batch)
        loss = criterion(batch, enhanced_image)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'unspuer_model.pth')