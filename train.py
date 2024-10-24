import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import argparse
import os
from ResNet import ResNet26, load_data  # 确保这里导入你的 ResNet26 和 load_data 函数

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
        img = torch.tensor(self.data[idx])  # 转为 Tensor
        label = self.labels[idx]

        # 确保图像形状为 (C, H, W)
        if img.shape[0] != 3:
            img = img.permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        return img, label


def load_model(weights_path=None):
    """加载模型并加载预训练权重（如果提供）。"""
    model = ResNet26()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict, strict=False)
    return model

def freeze_layers(model, block_indices=None):
    """冻结模型中的层，只允许指定的层进行训练。"""
    # 如果没有指定 block_indices，默认所有参数可训练
    if block_indices is None:
        for param in model.parameters():
            param.requires_grad = True
        print("All layers are set to be trainable.")
        return

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻指定的 block
    for block_index in block_indices:
        if block_index == 1:
            for param in model.layer1.parameters():
                param.requires_grad = True
        elif block_index == 2:
            for param in model.layer2.parameters():
                param.requires_grad = True
        elif block_index == 3:
            for param in model.layer3.parameters():
                param.requires_grad = True
        elif block_index == 4:
            for param in model.layer4.parameters():
                param.requires_grad = True
        else:
            print(f'Block {block_index} does not exist in the model.')

    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params}')

def train(model, train_loader, criterion, optimizer, device):
    """训练模型一个 epoch。"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    print(f'Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    return 100 * correct / total

def evaluate(model, test_loader, device):
    """在测试集上评估模型性能。"""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    return 100 * correct / total

def save_model(model, save_path, weights_name,lr):
    """保存模型权重。"""
    os.makedirs(save_path, exist_ok=True)
    save_full_path = os.path.join(save_path, f"{weights_name}_lr_{lr:.6f}.pth")  # 保留6位小数
    torch.save(model.state_dict(), save_full_path)
    print(f'Model saved to {save_full_path}')

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate ResNet-26 on CIFAR-10')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--test_data', type=str, required=True, help='Path to testing dataset')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights (optional)')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save model weights')
    parser.add_argument('--weights_name', type=str, default='resnet26_weights.pth', help='Filename for saved weights')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_block', type=int, nargs='+', default=None, help='Layer blocks to train (1-4)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据变换
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    train_dataset = CIFAR10Dataset(args.train_data, transform=transform)
    test_dataset = CIFAR10Dataset(args.test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    model = load_model(args.weights).to(device)

    # 冻结层
    freeze_layers(model, args.train_block)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_acc = 0
    best_model = model

    # 训练和评估
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}:')
        train(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        if best_acc<acc:
            # print("change")
            best_model = model
            best_acc = acc

    # 保存模型
    save_model(best_model, args.save_path, args.weights_name,args.lr)
    print(f"best accuracy on test data is {best_acc:.2f}")

if __name__ == '__main__':
    main()