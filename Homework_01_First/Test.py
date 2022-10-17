import numpy as np
import torch
from matplotlib import pyplot as plt

point_count = 100
# 是向量形式的
x_train = np.linspace(-3, 3, point_count)
y_train = np.sin(x_train) + np.random.uniform(-0.5, 0.5, point_count)

x_train = x_train.reshape((point_count, 1))
x_total = np.hstack((x_train, x_train**2, x_train**3))
x_train = torch.tensor(x_train).to(torch.float32)

y_train = y_train.reshape((point_count, 1))
y_train = torch.tensor(y_train).to(torch.float32)
