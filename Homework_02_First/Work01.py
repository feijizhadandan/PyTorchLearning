import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)

batch_size = 256
train_loader = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size, shuffle=False)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, din):
        din = din.view(-1, 784)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return dout


model = MLP()

lr = 0.1
num_epochs = 10
losss = []
accss = []


def train():
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    for epoch in range(num_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = lossfunc(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        losss.append(train_loss)
        test()
        print('Epoch:  {}  \tLoss: {:.6f}'.format(epoch + 1, train_loss))


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    accss.append(100.0 * correct / total)
    return 100.0 * correct / total


train()
print(f'正确率=: {accss[len(accss) - 1]}')
print(f'loss=: {losss[len(losss) - 1]}')
print(f'准确率-迭代次数变化图')
plt.plot(np.arange(num_epochs), accss)
print(f'loss-迭代次数变化图')
plt.plot(np.arange(num_epochs), losss)
