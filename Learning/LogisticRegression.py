import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

'''
    1、MNIST数据集的训练思路：
        012..9 分类，不能看做成有大小关系的分类
        输入一个值，输出每一个分类的概率 P(0)、P(1)、P(2)...P(9)，其中的最大值就可以看做是匹配结果
        - 下载Mnist数据集
            train_set = torchvision.datasets.MNIST(root='E:\Mnist\Train', train=True, download=True)
            test_set = torchvision.datasets.MNIST(root='E:\Mnist\Test', train=False, download=True)
    
    2、CIFAR-10数据集
    
    3、Logistic函数：保证输出值在 0-1 之间
'''

# 二分类问题
'''
    分析思路：
        1、y_hat 是表示 True 的概率，处于[0~1]之间；而 y 是已知数据，0/1
        2、loss 不再是一个度量值的结果，而是两个分布的差异(KL散度、交叉熵)
        3、loss 使用交叉熵求得，且越小越好
            y 确定的情况下，表达式中的 y_hat 越趋近 y，求得的 loss 越小
            对 loss 进行反向传播求出 loss/w，loss/b 的梯度，又因为梯度的反方向是 loss 减少的方向，因此往（梯度的反方向*学习率）更新即可
'''

'''
    1、准备数据集
'''
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

'''
    2、设计模型
'''


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 直接求出该值对应 [0~1] 中的分布值
        y_predict = torch.nn.functional.sigmoid(self.linear(x))
        return y_predict


# 实例化模型的可执行对象
model = LogisticRegressionModel()

'''
    3、设定损失函数和优化器
'''
# 损失函数 loss，使用交叉熵（之前使用平方和）
criterion = torch.nn.BCELoss(size_average=False)
# 优化器 仍然使用 SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

'''
    4、开始训练
'''
for epoch in range(1000):
    # 获得 y_hat 的 3*1 矩阵
    y_hat = model(x_data)
    # 根据 y_hat 和 y_data 计算出 loss(是一个标量，虽然应该也是一个 3*1 的矩阵，但是经过处理后得到了一个标量值)
    loss = criterion(y_hat, y_data)
    print("训练轮次：", epoch, "loss = ", loss.item())

    # 所有权重的梯度都归零
    optimizer.zero_grad()
    # 前面的梯度都归零后，就可以从最终的结果张量开始反向传播，自动求出前面的张量的梯度值
    loss.backward()
    # 更新 根据反向传播算出的梯度，乘上学习率进行权重的更新
    optimizer.step()

# 输出权重的训练结果
print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

# 测试数据集
x_test = torch.Tensor([[4.0]])
y_test_predict = model(x_test)
print("y_test_predict = ", y_test_predict)

'''
    绘制图形
'''
# 取 0~10 之间的200个点作为向量
x = np.linspace(0, 10, 200)
# 将 向量 转换成 200*1 的矩阵
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
# y_t 是张量，y_t.data 也是张量，y_t.data.numpy() 是 200*1 的矩阵
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('bbb')
plt.ylabel('aaa')
plt.grid()
plt.show()
