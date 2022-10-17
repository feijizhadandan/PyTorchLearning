import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

learning_rate = 0.01
w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_predict = forward(x)
    return (y_predict - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 得到的 loss 也是一个 tensor
        l = loss(x, y)
        # 反向传播求出计算图上所有 tensor 的梯度，并释放计算图
        l.backward()
        print('grad：', x, y, w.grad.item())
        w.data = w.data - learning_rate * w.grad.data

        # 清零 tensor w 中的梯度值，否则下次会进行累加
        w.grad.data.zero_()

    print("progress：", epoch, l.item())

print("predict (after training)", 4, forward(4).item())
