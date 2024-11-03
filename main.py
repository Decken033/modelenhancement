import sys
sys.path.append(r'f:\anaconda\envs\pytorch\lib\site-packages')
import os
import copy
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查 CUDA 可用性并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Count:", torch.cuda.device_count())

# 定义感知损失（Perceptual Loss）
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=11):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:feature_layer].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = self.criterion(input_features, target_features)
        return loss

# 定义简单的卷积自编码器
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 定义数据集类
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if not self.image_files:
            raise FileNotFoundError(f"在文件夹中未找到图像: {folder_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        target = transforms.functional.equalize(transforms.ToPILImage()(image))
        target = transforms.ToTensor()(target)

        return image, target

# 训练模型
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for images, targets in tqdm(dataloaders[phase]):
                if images is None or targets is None:
                    continue
                images = images.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if os.path.exists('best_model.pth'):
                    os.remove('best_model.pth')
                torch.save(best_model_wts, 'best_model.pth')
                print(f"New best model saved: best_model.pth")

    print(f"Training complete. Best val loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# 处理图像文件夹
def process_image_folder(folder_path, output_folder, model_path=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])

    dataset = ImageFolderDataset(folder_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    }

    model = AutoEncoder().to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        criterion = PerceptualLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = train_model(model, dataloaders, criterion, optimizer, num_epochs=20, device=device)

    os.makedirs(output_folder, exist_ok=True)

    # 测试最佳模型对单张图片的效果
    test_single_image(model, 'image.png', output_folder)

def test_single_image(model, image_path, output_folder):
    # 确保模型在评估模式
    model.eval()

    # 加载图像并预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0, 1).squeeze(0)  # 移除 batch 维度

    # 转换为图像并保存
    output_image = transforms.ToPILImage()(output.cpu())
    output_path = os.path.join(output_folder, 'enhanced_image.png')
    output_image.save(output_path, quality=95)
    print(f"Enhanced image saved as {output_path}")

    # 显示原始图像和增强后的图像的对比
    display_comparison(image, output_image)

def display_comparison(original_image, enhanced_image):
    # 显示对比图
    plt.figure(figsize=(10, 5))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    # 增强后的图像
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image")
    plt.imshow(enhanced_image)
    plt.axis('off')

    # 展示图像
    plt.show()

# 主函数
def main():
    folder_path = 'train'  # 替换为您的图像数据集文件夹路径
    output_folder = 'output'  # 处理后图像的输出文件夹
    model_path = None

    process_image_folder(folder_path, output_folder, model_path)

if __name__ == '__main__':
    main()
