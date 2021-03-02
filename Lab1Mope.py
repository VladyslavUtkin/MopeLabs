import numpy as np
from random import uniform
from prettytable import PrettyTable


table0 = PrettyTable()
table0.field_names = (["Студент", "Группа"])
name = "Уткін Владислав"
group = "ІО-92"
table0.add_row([name, group])
print(table0)

Min = 0
Max = 20

a0 = 1
a1 = 2
a2 = 3
a3 = 5

X = np.empty((8, 3), dtype=float)
Y = np.empty(8)
X0 = np.empty(3)
DX = np.empty(3)
XNormalized = np.empty((8, 3), dtype=float)
for i in range(8):
    for j in range(3):
        X[i, j] = uniform(Min, Max)
for i in range(8):
    Y[i] = a0 + a1 * X[i, 0] + a2 * X[i, 1] + a3 * X[i, 2]
for i in range(3):
    X0[i] = (X[:, i].max() + X[:, i].min()) / 2
    DX[i] = X[:, i].max() - X0[i]
Y_et = a0 + a1 * X0[0] + a2 * X0[1] + a3 * X0[2]
for i in range(8):
    for j in range(3):
        XNormalized[i, j] = (X[i, j] - X0[j]) / DX[j]
dY = 999999
number = -1
for i in range(8):
    if Y[i] - Y_et < dY and Y[i] - Y_et > 0:
        dY = Y[i] - Y_et
        number = i
Y2 = a0 + a1 * X[number, 0] + a2 * X[number, 1] + a3 * X[number, 2]
print("X:\n", X)
print("Y:\n", Y)
print("X0: \n", X0)
print("Y_et = ", Y_et)
print("XNormalized: \n", XNormalized.round(4))
print("number = ", number)
