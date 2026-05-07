import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. 读取原始数据并进行手动预处理 (Numpy 阶段)
# ==========================================
def load_raw_mnist():
    # 路径指向你之前保留的 4 个最原始的压缩包
    train_images_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
    train_labels_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
    test_images_path = './data/MNIST/raw/t10k-images-idx3-ubyte.gz'
    test_labels_path = './data/MNIST/raw/t10k-labels-idx1-ubyte.gz'

    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(16)  # 跳过 16 字节的头信息
            # 读出所有像素，转为 float32，并进行归一化 (0~1)
            data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
            return data.reshape(-1, 1, 28, 28)  # 转换为 PyTorch 要求的 [N, C, H, W]

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(8)   # 跳过 8 字节的头信息
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    # 加载训练集和测试集
    X_train = read_images(train_images_path)
    y_train = read_labels(train_labels_path)
    X_test = read_images(test_images_path)
    y_test = read_labels(test_labels_path)

    return X_train, y_train, X_test, y_test

# 提取 NumPy 数据
X_train, y_train, X_test, y_test = load_raw_mnist()

# 转换为 PyTorch 的 Tensor
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# 使用 TensorDataset 和 DataLoader 组装数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# ==========================================
# 2. 定义卷积神经网络 (CNN) 结构
# ==========================================
class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        # 卷积层 1：输入通道 1，输出通道 16，卷积核 3x3，填充 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸从 28x28 变成 14x14
        
        # 卷积层 2：输入 16，输出 32，卷积核 3x3，填充 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸从 14x14 变成 7x7
        
        # 全连接层：32张 7x7 的特征图 -> 10个数字分类
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平 (Flatten)
        x = self.fc(x)
        return x


# ==========================================
# 3. 准备设备、模型、损失函数与优化器
# ==========================================
# 如果电脑有英伟达显卡，会自动使用 GPU 加速，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

model = MNISTConvNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器


# ==========================================
# 4. 训练与测试循环
# ==========================================
def train(epoch):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播，PyTorch 自动帮你算好所有梯度！
        loss.backward()
        # 更新权重
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / 100:.6f}")
            running_loss = 0.0

def test():
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 测试阶段不需要计算梯度，省显存和时间
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 找出概率最大的指数
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


# ==========================================
# 5. 开始运行
# ==========================================
if __name__ == "__main__":
    epochs = 3  # 训练 3 轮
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()