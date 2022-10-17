import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# 激活函数使用 relu(), 不用 sigmoid()了
import torch.nn.functional as functional
import torch.optim as optim

transform = transforms.Compose([
    # 能将数据集中的图像转换成 1*28*28 的三维张量
    transforms.ToTensor(),
    # 将数据从 0~255 压缩到 0~1 之间的 0-1分布 (训练效果好)
    transforms.Normalize((0.1307,), (0.3081,))
])
'''
    下载数据集
'''
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class CnnNet(torch.nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        # 黑白图像，输入只有一个通道，输出通道数量就是卷积核个数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(7 * 7 * 64, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 10)
        self.dp = nn.Dropout(p=0.25)

    # 参数x：应当已经转换成 (batch_size, channel_cnt, width, height)格式的四维张量
    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        # 此时已经是 batch_size, 64, 7, 7, 线性变化前需要使用view转换成 batch_size, 64*7*7
        x = x.view(-1, 64 * 7 * 7)

        x = functional.relu(self.linear1(x))
        x = functional.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x


model = CnnNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

loss_list = []


def train(epoch_cnt):
    for epoch in range(epoch_cnt):
        # 用于计算这一轮的平均误差
        running_lose = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, targets = data
            # 将初始数据的张量迁移到 GPU 上
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # 该 loss 是这一批样本的总误差
            running_lose += loss.item()

        loss_list.append(running_lose / len(train_loader))
        print("训练轮次：%f, loss = %f" % (epoch, running_lose / len(train_loader)))


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print('测试集预测准确率: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    train(5)
    test()
    # loss-训练轮次 的关系图
    plt.plot(np.arange(5), loss_list)
    plt.show()
