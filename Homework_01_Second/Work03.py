import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# 激活函数使用 relu(), 不用 sigmoid()了
import torch.nn.functional as functional
import torch.optim as optim


batch_size = 64
transform = transforms.Compose([
    # 能将数据集中的图像转换成 1*28*28 的三维张量
    transforms.ToTensor(),
    # 将数据从 0~255 压缩到 0~1 之间的 0-1分布 (训练效果好)
    transforms.Normalize((0.1307, ), (0.3081, ))
])
'''
    下载数据集
'''
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


'''
    建立模型
'''
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        # 一组数据就是 28*28 个像素点数据，因此维度转换就是 28*28 -> 10(10种结果)
        self.func = nn.Linear(28*28, 10)

    def forward(self, x):
        # 将一组数据(28*28) 拼接成一个向量；-1表示将自动计算行数，在这里就是样本数量（batch_size）
        x = x.view(-1, 28*28)
        return self.func(x)


model = SoftmaxModel()
# CrossEntropyLoss = Softmax + NLLLose
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

loss_list = []
acc_list = []


def train(epoch_cnt):
    for epoch in range(epoch_cnt):
        running_lose = 0.0
        # batch_idx 是批次, data 是通过 DataLoader 获得的数据
        for batch_idx, data in enumerate(train_loader, 0):
            # 从数据中获取输入和输出值
            inputs, target = data
            optimizer.zero_grad()

            # 前向传播 + 反向传播 + 更新参数
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_lose += loss.item()

        loss_list.append(running_lose / batch_size)
        print("训练轮次：%f, loss = %f" % (epoch, running_lose / batch_size))


def test():
    correct = 0
    total = 0
    # 测试集不进行反向传播、优化
    with torch.no_grad():
        # 从 test 数据集中一批一批获取数据
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            # outputs.data的格式是 batch_size * 10 的矩阵
            # max 函数表示：沿着 output.data 矩阵的第一个维度（行是第0维度，列是第1维度），找出最大值下标，返回两个值：最大值 最大值下标
            _, predicted = torch.max(outputs.data, dim=1)
            # target 是一个 batch_size * 1 的矩阵，表示测试集结果矩阵
            total += target.size(0)
            # 张量间的比较运算
            correct += (predicted == target).sum().item()

    print('测试集预测准确率: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    train(5)
    test()
    # loss-训练轮次 的关系图
    plt.plot(np.arange(5), loss_list)
    plt.show()
