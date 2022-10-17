import torch

# 数据集，使用 Mini-Batch 梯度下降法，参数x为一维数据，一次批处理3条数据，因此设置大小为 3*1 的矩阵
x_data = torch.Tensor([
    [1.0], [2.0], [3.0]
])
y_data = torch.Tensor([
    [2.0], [4.0], [6.0]
])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 线性模型中输入和输出都是一维的
        self.linear = torch.nn.Linear(1, 1)

    # 前馈时所需要执行的计算
    def forward(self, x):
        y_predict = self.linear(x)
        return y_predict


# model 是我们构造的线性模型的实例化对象，是一个可执行对象，可以使用 model(x) 来调用其中的 __callable__ 函数
# model(x)：x 是批处理数据，是一个 n*1 的矩阵，可以求出一个 n*1 的矩阵“y”
model = LinearModel()

# 用于求损失函数的可执行对象，将从 model(x) 求出的结果矩阵“y hat” 和 已知结果"y"传入即可 criterion(yhat, y)
criterion = torch.nn.MSELoss(reduction='sum')

# 优化器 传入的参数是需要优化(用梯度下降SGD更新值)的 tensor
# model.parameters 用来获取 model 中的所有成员，并将其权重(参数)加入需要训练的范围内
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

'''
    训练过程：
        1、求出 y_hat
        2、求出 loss
        3、反向传播求出权重的梯度值
        4、根据梯度和学习率更新权重
        
    前三步都是在为最后一步训练做准备
'''
for epoch in range(1000):
    # 获得 y_hat 的 3*1 矩阵
    y_hat = model(x_data)
    # 根据 y_hat 和 y_data 计算出 loss(是一个标量，虽然应该也是一个 3*1 的矩阵，但是经过处理后得到了一个标量值)
    loss = criterion(y_hat, y_data)
    print("训练轮次：", epoch, "loss = ", loss)

    # 所有权重的梯度都归零
    optimizer.zero_grad()
    # 前面的梯度都归零后，就可以从最终的结果张量开始反向传播，自动求出前面的张量的梯度值
    loss.backward()
    # 更新 根据反向传播算出的梯度 和 学习率，进行权重的更新
    optimizer.step()

# 输出权重的训练结果
print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

# 测试数据集
x_test = torch.Tensor([[4.0]])
y_test_predict = model(x_test)
print("y_test_predict = ", y_test_predict)
