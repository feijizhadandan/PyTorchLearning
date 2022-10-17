import numpy as np
from matplotlib import pyplot as plt

# 求所有点的平均误差
def compute_error_for_given_points(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


# 通过当前的 w和b，根据 b_new = b_current - (learning_rate * b_gradient)，求出迭代后的 w和b
def get_new_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2 / n) * ((w_current * x) + b_current - y)
        w_gradient += (2 / n) * x * ((w_current * x) + b_current - y)

    b_new = b_current - (learning_rate * b_gradient)
    w_new = w_current - (learning_rate * w_gradient)
    return [b_new, w_new]


# 梯度下降算法开始器
def gradient_descent_runner(points, init_b, init_w, learning_rate, iteration_count):
    b = init_b
    w = init_w
    for i in range(iteration_count):
        b, w = get_new_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


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

    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    iteration_count = 1000

    print("Initial data: b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w, compute_error_for_given_points(initial_b, initial_w, points)))

    print("Start Running")

    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, iteration_count)

    print("Final data: b = {0}, w = {1}, error = {2}"
          .format(b, w, compute_error_for_given_points(b, w, points)))

    # 画出直线
    x = np.linspace(-3, 3, 100)
    y = w * x + b
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    run()
