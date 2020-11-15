import numpy as np

A = np.array([2, 1, 1])
B = np.array([1, 2, 1])
C = np.array([3, 4, 1])
D = (np.array([-1, -3, 1])).transpose()
mat = np.vstack((A, B, C))
mat = mat.transpose()


A2 = np.array([0, 1, 1])
B2 = np.array([5, 0, 1])
C2 = np.array([2, -5, 1])
D2 = (np.array([-1, -1, 1])).transpose()
mat2 = np.vstack((A2, B2, C2))
mat2 = mat2.transpose()

# I deo naivnog algoritma:
# Trazeni alfa, beta i gama za koje vazi:
# D = alfa * A + beta * B + gama * C
coefs = np.linalg.solve(mat, D)

P1 = np.vstack((A*coefs[0], B*coefs[1], C*coefs[2]))
P1 = P1.transpose()

# II deo naivnog algoritma:
# Trazeni alfa2, beta2, gama2 za koje vaze:
# D2 = alfa2 * A2 + beta2 * B2 + gama2 * C2
coefs2 = np.linalg.solve(mat2, D2)

P2 = np.vstack((A2*coefs2[0], B2*coefs2[1], C2*coefs2[2]))
P2 = P2.transpose()


# III deo naivnog algoritma:
# P = P2 * P1 (^-1)
P1 = np.linalg.inv(P1)
P = np.dot(P2, P1)
norm = 1/P[0][0]
print(norm)
P = np.array([np.array([dot[0]*norm, dot[1]*norm, dot[2]*norm]) for dot in P])
print(P)
