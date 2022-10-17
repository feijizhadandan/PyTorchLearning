import numpy as np
import torch
from matplotlib import pyplot as plt

point_count = 100
# 是向量形式的
x_train = np.linspace(-3, 3, point_count)
y_train = np.sin(x_train) + np.random.uniform(-0.5, 0.5, point_count)

x_train = x_train.reshape((point_count, 1))
# 转换成 n*1 矩阵
x_train = torch.tensor(x_train).to(torch.float32)

y_train = y_train.reshape((point_count, 1))
y_train = torch.tensor(y_train).to(torch.float32)

# plt.plot(x_train, y_train, 'ro')
# plt.show()

# 记录误差值
losses = []
# 记录 test 准确率
accuracy = []


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 传入的数据都是矩阵的形式
        y_predict = self.linear(x)
        return y_predict


model = LinearModel()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

epoch_cnt = 5000
for epoch in range(epoch_cnt):
    optimizer.zero_grad()

    y_hat = model(x_train)
    loss = criterion(y_hat, y_train)
    losses.append(loss.data.item() / 100.0)
    print("训练轮次：", epoch, "loss = ", loss)

    loss.backward()
    optimizer.step()


# 输出权重的训练结果
w = model.linear.weight.item()
b = model.linear.bias.item()
print("w = ", w)
print("b = ", b)

# x = np.linspace(-3, 3, point_count)
# y = w * x + b
# plt.plot(x_train, y_train, 'ro', x, y)
# plt.show()
plt.plot(np.arange(epoch_cnt), losses)
plt.show()
