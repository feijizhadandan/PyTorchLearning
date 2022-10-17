import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# 求所有点的平均误差
def compute_error_for_given_points(b, w1, w2, w3, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w1 * x + w2 * x**2 + w3 * x**3 + b)) ** 2
    return total_error / float(len(points))


def start():
    # 获取样本点
    num_observations = 100
    x = np.linspace(-3, 3, num_observations)
    y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)
    points = np.vstack((x, y))

    # 绘制样本点
    plt.scatter(x, y)

    x_feature1 = x.reshape(-1, 1)
    x_feature2 = x_feature1 ** 2
    x_feature3 = x_feature1 ** 3

    x_feature_total = np.hstack((x_feature1, x_feature2, x_feature3))

    # 调用线性回归函数
    lr = LinearRegression()
    lr.fit(x_feature_total, y)
    y_predict = lr.predict(x_feature_total)

    # 获取系数、截距
    w1, w2, w3 = lr.coef_
    b = lr.intercept_

    print("w1 = ", w1)
    print("w2 = ", w2)
    print("w3 = ", w3)
    print("b = ", b)
    print("loss = ", compute_error_for_given_points(b, w1, w2, w3, points))

    # 绘制线性结果
    plt.scatter(x, y)
    plt.plot(x, y_predict, color='r')
    plt.show()


if __name__ == '__main__':
    start()
