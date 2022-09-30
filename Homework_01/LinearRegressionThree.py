import numpy as np
from matplotlib import pyplot as plt


# 求所有点的平均误差
def compute_error_for_given_points(b, w1, w2, w3, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w1 * x + w2 * x**2 + w3 * x**3 + b)) ** 2
    return total_error / float(len(points))


# 通过当前的 w和b，根据 b_new = b_current - (learning_rate * b_gradient)，求出迭代后的 w和b
def get_new_gradient(b_current, w1_current, w2_current, w3_current, points, learning_rate):
    b_gradient = 0
    w1_gradient = 0
    w2_gradient = 0
    w3_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2 / n) * (w1_current * x + w2_current * x**2 + w3_current * x**3 + b_current - y)
        w1_gradient += (2 / n) * (w1_current * x + w2_current * x**2 + w3_current * x**3 + b_current - y) * x
        w2_gradient += (2 / n) * (w1_current * x + w2_current * x**2 + w3_current * x**3 + b_current - y) * x**2
        w3_gradient += (2 / n) * (w1_current * x + w2_current * x**2 + w3_current * x**3 + b_current - y) * x**3

    b_new = b_current - (learning_rate * b_gradient)
    w1_new = w1_current - (learning_rate * w1_gradient)
    w2_new = w2_current - (learning_rate * w2_gradient)
    w3_new = w3_current - (learning_rate * w3_gradient)
    return [b_new, w1_new, w2_new, w3_new]


# 梯度下降算法开始器
def gradient_descent_runner(points, init_b, init_w1, init_w2, init_w3, learning_rate, iteration_count):
    b = init_b
    w1 = init_w1
    w2 = init_w2
    w3 = init_w3
    for i in range(iteration_count):
        b, w1, w2, w3 = get_new_gradient(b, w1, w2, w3, np.array(points), learning_rate)
    return [b, w1, w2, w3]


# 获取 points 数据集函数
def get_points():
    num_observations = 100
    x = np.linspace(-3, 3, num_observations)
    y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)
    points = np.vstack((x, y))
    return points


def run():
    points = get_points()

    plt.scatter(points[0], points[1])

    learning_rate = 0.0000001
    initial_b = 0
    initial_w1 = 0
    initial_w2 = 0
    initial_w3 = 0
    iteration_count = 1000000

    print("Initial data: b = {0}, w1 = {1}, w2 = {2}, w3 = {3}, error = {4}"
          .format(initial_b, initial_w1, initial_w2, initial_w3,
                  compute_error_for_given_points(initial_b, initial_w1, initial_w2, initial_w3, points)))

    print("Start Running")

    [b, w1, w2, w3] = gradient_descent_runner(points, initial_b, initial_w1, initial_w2,
                                              initial_w3, learning_rate, iteration_count)

    print("Final data: b = {0}, w1 = {1}, w2 = {2}, w3 = {3},error = {4}"
          .format(b, w1, w2, w3, compute_error_for_given_points(b, w1, w2, w3, points)))

    # 画出直线
    x = np.linspace(-3, 3, 100)
    y = w1 * x + w2 * x**2 + -w3 * x**3 + b
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    run()
