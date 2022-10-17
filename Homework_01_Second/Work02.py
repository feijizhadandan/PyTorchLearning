import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

point_count = 100
# 是向量形式的
x_data = np.linspace(-3, 3, point_count)
y_data = np.sin(x_data) + np.random.uniform(-0.5, 0.5, point_count)

# 将 n 维向量转换成 n*1 矩阵
x_data = x_data.reshape((point_count, 1))
# 将 n * 1 矩阵转换为 n * 3 矩阵（因为是输入值是三维的）
x_train = np.hstack((x_data, x_data**2, x_data**3))
# 转换成 tensor
x_train = torch.tensor(x_train).to(torch.float32)

y_data = y_data.reshape((point_count, 1))
y_train = torch.tensor(y_data).to(torch.float32)


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        # 传入的数据都是矩阵的形式
        y_predict = self.linear(x)
        return y_predict


model = LinearModel()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(5000):
    optimizer.zero_grad()

    y_hat = model(x_train)
    loss = criterion(y_hat, y_train)
    print("训练轮次：", epoch, "loss = ", loss)

    loss.backward()
    optimizer.step()

# 输出权重的训练结果
w1 = model.linear.weight[:, 0].item()
w2 = model.linear.weight[:, 1].item()
w3 = model.linear.weight[:, 2].item()
b = model.linear.bias.data.item()
print("w1 = ", w1)
print("w2 = ", w2)
print("w3 = ", w3)
print("b = ", b)

x = np.linspace(-3, 3, point_count)
y = w1*x + w2*(x**2) + w3*(x**3) + b
plt.plot(x_data, y_data, 'ro', x, y)
plt.show()
