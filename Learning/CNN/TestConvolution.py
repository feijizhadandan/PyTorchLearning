import torch

'''
    对于图像的处理中，将 28*28 的图像转换成 1*784 的向量进行处理，会导致很多像素点的空间(相邻)关系信息丢失，因此需要运用到卷积运算
                
                栅格的通道数 * 宽度 * 高度 
    输入张量的格式：   n         w     h
    运算张量的格式：   n         w'    h'   m(卷积核数量)
    结果张量的格式：   m         w''   h''
    
'''

# 输入矩阵
input_matrix = [
    3, 4, 6, 5, 7,
    2, 4, 6, 8, 2,
    1, 6, 7, 8, 4,
    9, 7, 4, 6, 2,
    3, 7, 5, 4, 1
]

# 将矩阵转换成张量，并将维度设置为 1*1*5*5 -> batch_size, channel_cnt, width, height
input_matrix = torch.Tensor(input_matrix).view(1, 1, 5, 5)

# 卷积层：卷积核张量大小为 1*1*3*3：
#   kernel_size即卷积核的大小；padding表示输入矩阵需要扩张填充的距离(一般填充0)；bias表示卷积计算时需不需要加偏置值(本质是线性计算); stride表示步长
# padding：比如输入矩阵 5*5，padding=1 后会填充为 7*7，那么运算出来的结果就会是 5 *5,
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, stride=2)

# 设置卷积核张量(规格应该和上面卷积层运算器中设定的一样)
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
# 将卷积核张量赋值到卷积层中，初始化卷积计算的权重
conv_layer.weight.data = kernel.data

output_matrix = conv_layer(input_matrix)
print(output_matrix)

