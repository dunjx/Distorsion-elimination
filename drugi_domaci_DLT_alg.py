import numpy as np

n = int(input())

if n < 4:
    print("n mora da bude vece ili jednako 4!")
    exit(0)

mat = 0
mat2 = 0
print(mat)

print("Unesi n pocetnih tacaka:")
for i in range(n):
    a = int(input())
    b = int(input())
    c = int(input())
    if i == 0:
        mat = np.array([a, b, c])
    else:
        mat = np.vstack((mat, (np.array([a, b, c])).transpose()))

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

norm = 1/P2[0][0]
P2 = np.array([np.array([dot[0]*norm, dot[1]*norm, dot[2]*norm]) for dot in P2])
print(P2)
