import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)

batch_size = 256
train_loader = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size, shuffle=False)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dp = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(F.relu(x))

        x = self.conv2(x)

        x = self.pool2(F.relu(x))

        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        self.dp(x)
        return x


net = CNN()

lossfunc = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

accss = []
losss = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
epochs = 5

for epoch in range(epochs):
    train_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        inputs = data.to(device)
        labels = target.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = lossfunc(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    losss.append(train_loss)
    accss.append(100 * correct / total)
    print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    print('Epoch:  {}  \tTraining accs: {:.6f} %'.format(epoch + 1, 100 * correct / total))


def test():
    correct = 0
    train_loss = 0.0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in test_loader:
            inputs = data.to(device)
            labels = target.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            loss = lossfunc(outputs, labels)
            train_loss += loss.item()
            correct += (predicted == labels).sum().item()
    print('最终正确率为: %d %%' % (100 * correct / total))
    print('最终loss为: {:.6f}'.format(train_loss / total))

    return 100.0 * correct / total


test()

print(f'准确率随迭代次数的变化图为')
plt.plot(np.arange(epochs), accss)

print(f'损失loss随迭代次数的变化图为')
plt.plot(np.arange(epochs), losss)
