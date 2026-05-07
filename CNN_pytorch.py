import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


#get data

def load_raw_mnist():

    train_images_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
    train_labels_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
    test_images_path = './data/MNIST/raw/t10k-images-idx3-ubyte.gz'
    test_labels_path = './data/MNIST/raw/t10k-labels-idx1-ubyte.gz'

    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(16)  

            data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
            return data.reshape(-1, 1, 28, 28)  # [N, C, H, W]

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(8)   
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    X_train = read_images(train_images_path)
    y_train = read_labels(train_labels_path)
    X_test = read_images(test_images_path)
    y_test = read_labels(test_labels_path)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_raw_mnist()


X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # 卷积层 2：输入 16，输出 32，卷积核 3x3，填充 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f": {device}")

model = MNISTConvNet().to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器


def train(epoch):
    model.train() 
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / 100:.6f}")
            running_loss = 0.0

def test():
    model.eval() 
    test_loss = 0
    correct = 0
    with torch.no_grad(): 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


if __name__ == "__main__":
    epochs = 3 
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
