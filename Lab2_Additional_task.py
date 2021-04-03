#Перероблена лабораторна робота №2
from random import randint
import math
import numpy as np
class Lab2:
    def __init__(self):
        #Задані дані
        self.y_max = (30 - 19) * 10
        self.y_min = (20 - 19) * 10
        self.X1_min = 15
        self.X1_max = 45
        self.X2_min = -70
        self.X2_max = -10
        self.N = 5

        #Значення критерію Романовського за різних довірчих ймовірностей p кількостях дослідів m
        self.p_list = (0.99, 0.98, 0.95, 0.90)
        self.table_RKR = {2: (1.73, 1.72, 1.71, 1.69),
                          6: (2.16, 2.13, 2.10, 2.00),
                          8: (2.43, 4.37, 2.27, 2.17),
                          10: (2.62, 2.54, 2.41, 2.29),
                          12: (2.75, 2.66, 2.52, 2.39),
                          15: (2.9, 2.8, 2.64, 2.49),
                          20: (3.08, 2.96, 2.78, 2.62)}

        self.generate_y()
        self.exp()
        self.calculate()
        self.print()

    def generate_y(self):
        # Заповнимо матрицю планування для m=5
        self.y = [[randint(self.y_min, self.y_max) for j in range(self.N)] for i in range(3)]
        self.xn = [[-1, 1, -1], [-1, -1, 1]]
        print("x1min = {},x1max = {} \nx2min = {}, x2max = {}\nxn{} ".format(self.X1_min, self.X1_max, self.X2_min, self.X2_max, self.xn))
        print("y = ")
        for row in self.y:
            print(' | '.join([str(elem) for elem in row]))


    def exp(self):
        #---Перевірка однорідності дисперсії за критерієм Романовського---
        #1.Знайдемо середнє значення функції відгуку в рядку:
        self.average_Y1 = sum(self.y[0][j] for j in range(self.N))/self.N
        self.average_Y2 = sum(self.y[1][j] for j in range(self.N))/self.N
        self.average_Y3 = sum(self.y[2][j] for j in range(self.N))/self.N
        #2.Знайдемо дисперсії по рядках:
        self.D_Y1 = sum([(j - self.average_Y1) ** 2 for j in self.y[0]]) / self.N
        self.D_Y2 = sum([(j - self.average_Y2) ** 2 for j in self.y[1]]) / self.N
        self.D_Y3 = sum([(j - self.average_Y3) ** 2 for j in self.y[2]]) / self.N
        #3.Обчислимо основне відхилення:
        self.main_deviation = math.sqrt((2 * (2 * self.N - 2)) / (self.N * (self.N - 4)))
        #4.Обчислимо Fuv:
        self.Fuv_1 = self.D_Y1 / self.D_Y2
        self.Fuv_2 = self.D_Y3 / self.D_Y1
        self.Fuv_3 = self.D_Y3 / self.D_Y2
        #4.Обчислимо TETAuv:
        self.TETAuv_1 = ((self.N - 2) / self.N) * self.Fuv_1
        self.TETAuv_2 = ((self.N - 2) / self.N) * self.Fuv_2
        self.TETAuv_3 = ((self.N - 2) / self.N) * self.Fuv_3
        #6.Обчислимо Ruv:
        self.Ruv_1 = abs(self.TETAuv_1 - 1) / self.main_deviation
        self.Ruv_2 = abs(self.TETAuv_2 - 1) / self.main_deviation
        self.Ruv_3 = abs(self.TETAuv_3 - 1) / self.main_deviation

        if not self.check_homogeneity():
            print(f'\n Дісперсія неоднорідна! Змінимо m={self.N} to m={self.N + 1}\n')
            self.N += 1
            self.add()

    def add(self):
        for i in range(3):
            self.y[i].append(randint(self.y_min, self.y_max))
        self.exp()

    #---Перевірка однорідності дисперсії за критерієм Романовського---
    def check_homogeneity(self):
        m = min(self.table_RKR, key=lambda x: abs(x - self.N))
        p = 0
        for ruv in (self.Ruv_1, self.Ruv_2, self.Ruv_3):
            if ruv > self.table_RKR[m][0]:
                return False
        for rkr in range(len(self.table_RKR[m])):
            if ruv < self.table_RKR[m][rkr]:
                p = rkr
        temp = self.table_RKR[m][p]
        global p2
        p2 = self.p_list[p]
        global item_table
        item_table = temp
        return True

    def calculate(self):
        # Розрахуємо нормованих коефіцієнтів рівняння регресії.
        self.mx1 = sum(self.xn[0]) / 3
        self.mx2 = sum(self.xn[1]) / 3

        self.my = (self.average_Y1 + self.average_Y2 + self.average_Y3) / 3

        self.a1 = sum([i ** 2 for i in self.xn[0]]) / 3
        self.a2 = sum(self.xn[0][i] * self.xn[1][i] for i in range(3)) / 3
        self.a3 = sum([i ** 2 for i in self.xn[1]]) / 3
        self.a11 = (self.xn[0][0] * self.average_Y1 + self.xn[0][1] * self.average_Y2 + self.xn[0][2] * self.average_Y3) / 3
        self.a22 = (self.xn[1][0] * self.average_Y1 + self.xn[1][1] * self.average_Y2 + self.xn[1][2] * self.average_Y3) / 3

        self.B0 = np.linalg.det([[self.my, self.mx1, self.mx2], [self.a11, self.a1, self.a2], [self.a22, self.a2, self.a3]]) / (
            np.linalg.det([[1, self.mx1, self.mx2], [self.mx1, self.a1, self.a2], [self.mx2, self.a2, self.a3]]))
        self.B1 = np.linalg.det([[1, self.my, self.mx2], [self.mx1, self.a11, self.a2], [self.mx2, self.a22, self.a3]]) / (
            np.linalg.det([[1, self.mx1, self.mx2], [self.mx1, self.a1, self.a2], [self.mx2, self.a2, self.a3]]))
        self.B2 = np.linalg.det([[1, self.mx1, self.my], [self.mx1, self.a1, self.a11], [self.mx2, self.a2, self.a22]]) / (
            np.linalg.det([[1, self.mx1, self.mx2], [self.mx1, self.a1, self.a2], [self.mx2, self.a2, self.a3]]))

        # Проводимо натуралізацію коефіцієнтів:
        self.delta_x1 = math.fabs(self.X1_max - self.X1_min) / 2
        self.delta_x2 = math.fabs(self.X2_max - self.X2_min) / 2
        self.x10 = (self.X1_max + self.X1_min) / 2
        self.x20 = (self.X2_max + self.X2_min) / 2
        self.a2_0 = self.B0 - (self.B1 * (self.x10 / self.delta_x1)) - (self.B2 * (self.x20 / self.delta_x2))
        self.a2_1 = self.B1 / self.delta_x1
        self.a2_2 = self.B2 / self.delta_x2

    def print(self):
        print("Остаточні дані після перевірок : Y_max = {}  Y_min = {}  X1_min = {}  X1_max = {}  X2_min = {}  X2_max = {}".format(
            self.y_max, self.y_min, self.X1_min, self.X1_max, self.X2_min, self.X2_max))
        print("Матриця планування для m = {}".format(self.N))
        print("y = ")
        for row in self.y:
            print(' | '.join([str(elem) for elem in row]))


        print("1. Cереднє значення функції відгуку в рядку: Y1 = {}  Y2 = {}  Y3 = {}".format(self.average_Y1, self.average_Y2,
                                                                                              self.average_Y3))
        print("2. Значення дисперсії по рядках: σ²(Y1) = {}  σ²(Y2) = {}  σ²(Y3) = {}".format("%.2f" % self.D_Y1,
                                                                                              "%.2f" % self.D_Y2,
                                                                                              "%.2f" % self.D_Y3))
        print("3. Основне відхилення σθ: {}".format("%.2f" % self.main_deviation))
        print("4. Обчислюємо Fuv: Fuv_1 = {}  Fuv_2 = {}  Fuv_3 = {}".format("%.2f" % self.Fuv_1, "%.2f" % self.Fuv_2,
                                                                             "%.2f" % self.Fuv_3))
        print("5. Обчислюємо θuv: θ_uv1 = {}  θ_uv2 = {}  θ_uv3 = {}".format("%.2f" % self.TETAuv_1, "%.2f" % self.TETAuv_2,
                                                                             "%.2f" % self.TETAuv_3))
        print("6. Обчислюємо Ruv: Ruv_1 = {}  Ruv_2 = {}  Ruv_3 = {}".format("%.2f" % self.Ruv_1, "%.2f" % self.Ruv_2,
                                                                             "%.2f" % self.Ruv_3))
        print("Ruv1 = {} < Rкр = {}".format("%.2f" % self.Ruv_1, item_table))
        print("Ruv2 = {} < Rкр = {}".format("%.2f" % self.Ruv_2, item_table))
        print("Ruv3 = {} < Rкр = {}".format("%.2f" % self.Ruv_3, item_table))
        print("Однорідність дисперсій підтверджується з ймовірністю p = {} !".format(p2))
        print("mx1 = {}  mx2 = {}  my = {}".format("%.2f" % self.mx1, "%.2f" % self.mx2, "%.2f" % self.my))
        print("a1 = {}  a2 = {}  a3 = {}".format("%.2f" % self.a1, "%.2f" % self.a2, "%.2f" % self.a3))
        print("a11 = {}  a22 = {}     =>    B0 = {}  B1 = {}  B2 = {}".format("%.2f" % self.a11, "%.2f" % self.a22, "%.2f" % self.B0,
                                                                              "%.2f" % self.B1, "%.2f" % self.B2))
        print("Нормоване рівняння регресії : y = {} + ({})*x1 + ({})*x2 ".format("%.2f" % self.B0, "%.2f" % self.B1, "%.2f" % self.B2))
        print("Δx1 = {}  Δx2 = {}  X10 = {}  X20 = {}".format(self.delta_x1, self.delta_x2, self.x10, self.x20))
        print("a0 = {}  a1 = {}  a2 = {}".format("%.2f" % self.a2_0, "%.2f" % self.a2_1, "%.2f" % self.a2_2))
        print("Натуралізоване рівняння регресії: y = {} + ({})*x1 + ({})*x2 ".format("%.2f" % self.a2_0, "%.2f" % self.a2_1,
                                                                                     "%.2f" % self.a2_2))
Lab2()