import numpy as np


# 创建一维数组a
a = np.array([4, 5, 6])
# 输出a的类型
print(a.dtype)
# 输出a的各维度大小
print(a.shape)
# 输出a的第一个元素
print(a[0])


print("====================分隔符====================")

# 创建二维数组b
b = np.array([[4, 5, 6], [1, 2, 3]])
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 1])


print("====================分隔符====================")
# 建立矩阵
c = np.zeros((3, 3))
print(c)
d = np.ones((4, 5))
print(d)
e = np.identity(4)
print(e)


print("====================分隔符====================")
# 建立数组f
f = np.arange(12)
print(f, f.shape)
f = f.reshape((3, 4))
print(f, f.shape)
print(f[1, :])
print(f[:, 2:])
print(f[2, -1])
