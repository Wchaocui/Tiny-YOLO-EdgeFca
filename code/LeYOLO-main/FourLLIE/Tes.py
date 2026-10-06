import torch
from FourLLIE.models.archs import FourLLIE
from torchvision.transforms import Compose, ToTensor, Resize,Normalize
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os


# 设置环境变量
os.environ['NUMEXPR_MAX_THREADS'] = '24'
# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model = FourLLIE.FourLLIE(nf=64).to(device)
model.load_state_dict(torch.load('unspuer_model.pth', map_location=device))
model.eval()  # 设置为评估模式
class TestImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 测试图像路径
test_image_paths = ['img000126.jpg']

# 定义变换
transform = Compose([
    ToTensor(),Resize(256),  # Resize images
    Normalize((0.1541, 0.1554, 0.1596), (0.0420, 0.0403, 0.0368))
])

# 初始化测试数据集
test_dataset = TestImageDataset(test_image_paths, transform=transform)

# 初始化数据加载器
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# 测试模型
for i, image in enumerate(test_dataloader):
    image = image.to(device)

    # 前向传播
    with torch.no_grad():
        enhanced_image, _, _, _ = model(image)


    # 反归一化，以便显示和保存图像
    def unnormalize(tensor):
        tensor = tensor * 0.5 + 0.5
        return tensor


    original_image = unnormalize(image)
    enhanced_image = unnormalize(enhanced_image)

    # 保存图像
    TF.to_pil_image(original_image.squeeze(0)).save(f'original_image_{i}.png')
    TF.to_pil_image(enhanced_image.squeeze(0)).save(f'enhanced_image_{i}.png')

    # 显示图像
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(TF.to_pil_image(original_image.squeeze(0)))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Enhanced Image')
    plt.imshow(TF.to_pil_image(enhanced_image.squeeze(0)))
    plt.axis('off')

    plt.show()