import torch

# 输入矩阵
input_matrix = [
    3, 4, 6, 5, 7,
    2, 4, 6, 4, 2,
    1, 6, 7, 8, 4,
    9, 7, 4, 6, 2,
    3, 7, 5, 4, 1
]

input_matrix = torch.Tensor(input_matrix).view(1, 1, 5, 5)

# 池化层：选出一块区域内的最大值
pool_layer = torch.nn.MaxPool2d(kernel_size=3)

output_matrix = pool_layer(input_matrix)
print(output_matrix)
