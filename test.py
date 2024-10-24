import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import argparse

# 自定义数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        self.data = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # 归一化
        self.labels = batch[b'labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx])  # 确保转换为 Tensor
        label = self.labels[idx]
        
        # 检查是否需要调整张量形状
        if img.shape[0] != 3:  
            img = img.permute(2, 0, 1)  # 调整为 (C, H, W)

        if self.transform:
            img = self.transform(img)
        return img, label

# 定义 ResNet-26 的结构
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet26(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet26, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 4, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 4, stride=2)
        self.layer4 = self._make_layer(256, 512, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def load_model(weights_path):
    """加载模型和权重"""
    model = ResNet26()
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict, strict=False)  # 允许部分不匹配
    model.eval()  # 设置为评估模式
    return model

def evaluate(model, dataloader, device):
    """在测试集上进行评估"""
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='Test ResNet-26 on CIFAR-10 dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to test dataset (e.g., test_raw/test_raw_batch.pkl)')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (e.g., resnet26_weights.pth)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    args = parser.parse_args()

    # 设备配置 (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据变换
    transform = transforms.Compose([
        # transforms.ToTensor(),  # 转换为 Tensor，且形状为 (C, H, W)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正则化
    ])

    # 加载数据集
    dataset = CIFAR10Dataset(args.data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    model = load_model(args.weights)

    # 评估模型
    evaluate(model, dataloader, device)

if __name__ == '__main__':
    main()