import sys
import os
import copy
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval().to(device)
        self.vgg_layers = nn.ModuleList([
            vgg[:4],  # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16],  # relu3_3
        ])

        for param in self.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        content_loss = 0.0
        style_loss = 0.0

        for layer in self.vgg_layers:
            x = layer(x)
            y = layer(y)
            content_loss += self.mse_loss(x, y)

            # Add style loss using Gram matrices
            b, c, h, w = x.size()
            x_flat = x.view(b, c, -1)
            y_flat = y.view(b, c, -1)

            x_gram = torch.bmm(x_flat, x_flat.transpose(1, 2))
            y_gram = torch.bmm(y_flat, y_flat.transpose(1, 2))

            style_loss += self.mse_loss(x_gram, y_gram) / (c * h * w)

        return content_loss + 0.1 * style_loss


class EnhancedAutoEncoder(nn.Module):
    def __init__(self):
        super(EnhancedAutoEncoder, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(6)]
        )

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 3, kernel_size=7, padding=3)

        # Normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # Global contrast enhancement branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 256)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.enc_conv1(x)))
        x2 = F.relu(self.bn2(self.enc_conv2(x1)))
        x3 = F.relu(self.bn3(self.enc_conv3(x2)))

        # Residual blocks
        res = self.res_blocks(x3)

        # Global feature branch
        global_feat = self.global_pool(res)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = self.fc2(global_feat).view(global_feat.size(0), -1, 1, 1)

        # Combine global and local features
        enhanced = res * torch.sigmoid(global_feat)

        # Decoder
        d1 = F.relu(self.dec_conv1(enhanced))
        d2 = F.relu(self.dec_conv2(d1))
        out = torch.sigmoid(self.dec_conv3(d2))

        return out


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            # Create enhanced target using traditional image processing
            target = transforms.functional.equalize(transforms.ToPILImage()(image))
            target = transforms.ToTensor()(target)

            # Apply additional enhancements to target
            target = torch.clamp(target * 1.2, 0, 1)  # Increase contrast

            return image, target
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_perceptual_loss = 0.0

            for images, targets in tqdm(dataloaders[phase]):
                if images is None or targets is None:
                    continue

                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)

                    # Combine L1 loss with perceptual loss
                    l1_loss = F.l1_loss(outputs, targets)
                    perceptual_loss = criterion(outputs, targets)
                    loss = l1_loss + 0.1 * perceptual_loss

                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_perceptual_loss += perceptual_loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_perceptual_loss = running_perceptual_loss / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f}, Perceptual Loss: {epoch_perceptual_loss:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'best_model.pth')
                    print(f"New best model saved")

        print()

    print(f"Best val loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def process_image_folder(folder_path, output_folder, model_path=None):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Increased image size
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),  # Added rotation augmentation
        transforms.ToTensor(),
    ])

    dataset = ImageFolderDataset(folder_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    }

    model = EnhancedAutoEncoder().to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        criterion = PerceptualLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        model = train_model(model, dataloaders, criterion, optimizer, num_epochs=30, device=device)

    os.makedirs(output_folder, exist_ok=True)
    test_single_image(model, 'image.png', output_folder)


def test_single_image(model, image_path, output_folder):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0, 1).squeeze(0)

    output_image = transforms.ToPILImage()(output.cpu())
    output_path = os.path.join(output_folder, 'enhanced_image.png')
    output_image.save(output_path, quality=95)
    print(f"Enhanced image saved as {output_path}")

    display_comparison(image, output_image, output_folder)

def display_comparison(original_image, enhanced_image, output_folder):
    # 统一两张图片的尺寸
    target_size = (512, 512)
    original_image = original_image.resize(target_size)
    enhanced_image = enhanced_image.resize(target_size)

    # 创建一个新的图像，将两张图片并排拼接
    combined_width = original_image.width + enhanced_image.width
    combined_image = Image.new('RGB', (combined_width, target_size[1]))

    # 将原图和增强图粘贴到组合图像中
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(enhanced_image, (original_image.width, 0))

    # 保存组合后的图像
    output_path = os.path.join(output_folder, 'comparison_image.png')
    combined_image.save(output_path, quality=95)
    print(f"对比图已保存为 {output_path}")

    # 使用 matplotlib 显示组合后的图像
    plt.figure(figsize=(10, 5))
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()

def main():
    folder_path = 'train'
    output_folder = 'output'
    model_path = None
    process_image_folder(folder_path, output_folder, model_path)


if __name__ == '__main__':
    main()