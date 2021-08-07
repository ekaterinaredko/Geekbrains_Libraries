#1
import numpy as np
a = np.arange(12, 24)
print(a)
#2
a.reshape(3, 4)
a.reshape(4, 3)
a.reshape(6, 2)
a.reshape(2, 6)
a.resize(12, 1)
#3
a.reshape(-1, 3)
a.reshape(-1, 6)
a.reshape(-1, 4)
a.reshape(-1, 2)
a.reshape(12, -1)
#4
a.ndim
#Ответ: нет
#5
a = np.random.randn(3, 4)
print(a)
b = a.flatten()
print(b)
a.size == b.size
#6
a = np.arange(20, 0, -2)
print(a)
#7
b = np.arange(20, 1, -2)
print(b)
np.array_equal(a, b)
#Ответ: нет разницы
#8
a = np.zeros((3, 2))
b = np.ones((2, 2))
v = np.concatenate((a, b), axis = 0)
print(v)
v.size
#9
a = np.arange(0, 12)
print(a)
A = a.reshape(4, 3)
print(A)
At = A.transpose()
print(At)
B = A@At
print(B)
B.size
np.linalg.det(B)
#10
np.random.seed(42)
#11
c = np.arange(0, 16)
print(c)
#12
C = c.reshape((4, 4))
print(C)
D = 10 * C + B
print(D)
np.linalg.det(D)
np.linalg.matrix_rank(D)
D_inv = np.linalg.inv(D)
print(D_inv)
#13
mask_positive = D_inv > 0
mask_negative = D_inv < 0
D_inv[mask_positive] = 1
D_inv[mask_negative] = 0
print(D_inv)
E = np.where(D_inv, B, C)
print(E)