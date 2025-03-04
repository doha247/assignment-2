import numpy as np


X = np.array([[0.05, 0.10]])  # 1x2

w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55


Y = np.array([[0.01, 0.99]])

alpha = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

net_h1 = X[0, 0] * w1 + X[0, 1] * w3
net_h2 = X[0, 0] * w2 + X[0, 1] * w4

out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

net_o1 = out_h1 * w5 + out_h2 * w7
net_o2 = out_h1 * w6 + out_h2 * w8

out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)

error_o1 = (out_o1 - Y[0, 0])
error_o2 = (out_o2 - Y[0, 1])

delta_o1 = error_o1 * sigmoid_derivative(out_o1)
delta_o2 = error_o2 * sigmoid_derivative(out_o2)

partial_Etotal_w5 = delta_o1 * out_h1
dw5 = alpha * partial_Etotal_w5
partial_Etotal_w6 = delta_o2 * out_h1
dw6 = alpha * partial_Etotal_w6
partial_Etotal_w7 = delta_o1 * out_h2
dw7 = alpha * partial_Etotal_w7
partial_Etotal_w8 = delta_o2 * out_h2
dw8 = alpha * partial_Etotal_w8


error_h1 = delta_o1 * w5 + delta_o2 * w6
error_h2 = delta_o1 * w7 + delta_o2 * w8

delta_h1 = error_h1 * sigmoid_derivative(out_h1)
delta_h2 = error_h2 * sigmoid_derivative(out_h2)


partial_Etotal_w1 = delta_h1 * X[0, 0]
dw1 = alpha * partial_Etotal_w1
partial_Etotal_w2 = delta_h2 * X[0, 0]
dw2 = alpha * partial_Etotal_w2
partial_Etotal_w3 = delta_h1 * X[0, 1]
dw3 = alpha * partial_Etotal_w3
partial_Etotal_w4 = delta_h2 * X[0, 1]
dw4 = alpha * partial_Etotal_w4

w1 -= dw1
w2 -= dw2
w3 -= dw3
w4 -= dw4
w5 -= dw5
w6 -= dw6
w7 -= dw7
w8 -= dw8


print(f"Updated w1: {w1}")
print(f"Updated w2: {w2}")
print(f"Updated w3: {w3}")
print(f"Updated w4: {w4}")
print(f"Updated w5: {w5}")
print(f"Updated w6: {w6}")
print(f"Updated w7: {w7}")
print(f"Updated w8: {w8}")
