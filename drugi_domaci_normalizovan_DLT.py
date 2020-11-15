import numpy as np
from functools import reduce

n = int(input())

if n < 4:
    print("n mora da bude vece ili jednako 4!")
    exit(0)

mat = 0
mat2 = 0

print("Unesi n pocetnih tacaka:")
for i in range(n):
    a = int(input())
    b = int(input())
    c = int(input())
    if i == 0:
        mat = np.array([a/c, b/c, 1])
    else:
        mat = np.vstack((mat, (np.array([a/c, b/c, 1])).transpose()))

print("Unesi n krajnjih tacaka:")
for i in range(n):
    a = int(input())
    b = int(input())
    c = int(input())
    if i == 0:
        mat2 = np.array([a, b, c])
    else:
        mat2 = np.vstack((mat2, (np.array([a, b, c])).transpose()))

print(mat)
print(mat2)

def normalize(mat):
    # Teziste pocetnih tacaka:
    mat_afine = np.array([np.array([dot[0]/dot[2], dot[1]/dot[2], 1]) for dot in mat])

    c = tuple(reduce(lambda x, y: x + y, map(lambda x: np.array(x), mat_afine)))
    c = np.array(c) / len(mat_afine)

    G = np.array([[1, 0, -c[0]],
                  [0, 1, -c[1]],
                  [0, 0, 1]])

    # Homotetija S:
    mat_new = [G.dot(np.array(dot)) for dot in mat_afine]
    distances = [np.sqrt((dot[0]*dot[0]) + (dot[1]*dot[1])) for dot in mat_new]

    mat_afine_new = [G.dot(np.array(dot)) for dot in mat_afine]

    lmd = sum(distances) / len(mat_afine_new)

    k = np.sqrt(2) / lmd

    S = np.array([[k, 0, 0],
                  [0, k, 0],
                  [0, 0, 1]])

    # Matrica translacije:
    T = np.dot(S, G)
    return T, [T.dot(np.array(dot)) for dot in mat]


T, mat = normalize(mat)
T2, mat2 = normalize(mat2)

P = 0

for i in range(n):
    pom_arr = mat[i]
    pom_arr2 = mat2[i]

    if i == 0:
        P = np.vstack((
            np.array([0, 0, 0,
                      -pom_arr2[2]*pom_arr[0], -pom_arr2[2]*pom_arr[1], -pom_arr2[2]*pom_arr[2],
                      pom_arr2[1]*pom_arr[0], pom_arr2[1]*pom_arr[1], pom_arr2[1]*pom_arr[2]]),
            np.array([pom_arr2[2]*pom_arr[0], pom_arr2[2]*pom_arr[1], pom_arr2[2]*pom_arr[2],
                      0, 0, 0,
                      -pom_arr2[0]*pom_arr[0], -pom_arr2[0]*pom_arr[1], -pom_arr2[0]*pom_arr[2]])))
    else:
        P = np.vstack((P, np.array([0, 0, 0,
                      -pom_arr2[2] * pom_arr[0], -pom_arr2[2] * pom_arr[1], -pom_arr2[2] * pom_arr[2],
                      pom_arr2[1] * pom_arr[0], pom_arr2[1] * pom_arr[1], pom_arr2[1] * pom_arr[2]]),
            np.array([pom_arr2[2] * pom_arr[0], pom_arr2[2] * pom_arr[1], pom_arr2[2] * pom_arr[2],
                      0, 0, 0,
                      -pom_arr2[0] * pom_arr[0], -pom_arr2[0] * pom_arr[1], -pom_arr2[0] * pom_arr[2]])))

_, _, V = np.linalg.svd(P)
V = np.transpose(V)
P2 = V[:, -1]
P2 = P2.reshape(3, 3)
P2 = np.dot(np.dot(np.linalg.inv(T2), P2), T)
norm = 1/P2[0][0]
P2 = np.array([np.array([dot[0]*norm, dot[1]*norm, dot[2]*norm]) for dot in P2])
print(P2)

