import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)
Closs = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.softmax(x)
        return x


model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
losss = []
accss = []


def train():
    for i, data in enumerate(train_iter):
        inputs, labels = data
        out = model(inputs)
        labels = labels.reshape(-1, 1)
        one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)
        loss = Closs(out, one_hot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test():
    correct = 0
    curLoss = 0
    for i, data in enumerate(test_iter):
        inputs, labels = data
        out = model(inputs)
        labelTmp = labels.reshape(-1, 1)
        one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labelTmp, 1)
        curLoss += Closs(out, one_hot).sum().item()
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum().item()
    print("acc:{0}".format(correct / len(mnist_test)), "loss:{0}".format(curLoss / len(mnist_test)))
    accss.append(correct / len(mnist_test))


losss.append(curLoss / len(mnist_test))
for epoch in range(20):
    print("epoch:", epoch)
    train()
test()
print(f'最终正确率为: {accss[len(accss) - 1]}')
print(f'最终loss为: {losss[len(losss) - 1]}')
print(f'准确率随迭代次数的变化图为')
plt.plot(np.arange(epoch + 1), accss)
print(f'损失loss随迭代次数的变化图为')
plt.plot(np.arange(epoch + 1), losss)