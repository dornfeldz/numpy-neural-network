import numpy as np
import math

normalized_data = [
    [[0.5556, 0.8471, 0.3636], 1],
    [[0.3333, 0.7647, 0.6364], 0],
    [[0.7778, 0.9412, 0.2727], 1],
    [[0.4444, 0.7059, 0.8182], 0],
    [[1.0, 1.0, 0.1818], 1],
    [[0.2778, 0.6824, 0.9091], 0],
    [[0.6667, 0.8235, 0.4545], 1],
    [[0.2222, 0.5882, 1.0], 0],
    [[0.8889, 0.9176, 0.3273], 1],
    [[0.3889, 0.7294, 0.7273], 0]
]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

w = np.random.random(4) * 0.8 - 0.4

for i in range(30000):
    err_sum = 0.0
    for inp, out in normalized_data:
        x = np.array(inp + [1.0])
        y = sigmoid(np.dot(x, w))
        err = out - y
        delta = err * d_sigmoid(y)
        w += 2.2 * delta * x
        err_sum += err ** 2
print(i, err_sum)

def predict(x):
    x_new = np.array(x + [1.0])
    y_pred = sigmoid(np.dot(x_new, w))
    return y_pred

print(predict([20/90, 250/850, 20/55]))
print(predict([90/90, 850/850, 1/55]))
print(predict([3/90, 20/850, 55/55]))

print("Hello, world!")