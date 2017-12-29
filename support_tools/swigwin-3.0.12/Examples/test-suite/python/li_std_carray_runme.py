from li_std_carray import *


v3 = Vector3()
for i in range(0, len(v3)):
    v3[i] = i

i = 0
for d in v3:
    if d != i:
        raise RuntimeError
    i = i + 1


m3 = Matrix3()

for i in range(0, len(m3)):
    v3 = m3[i]
    for j in range(0, len(v3)):
        v3[j] = i + j

i = 0
for v3 in m3:
    j = 0
    for d in v3:
        if d != i + j:
            raise RuntimeError
        j = j + 1
        pass
    i = i + 1
    pass

for i in range(0, len(m3)):
    for j in range(0, len(m3)):
        if m3[i][j] != i + j:
            raise RuntimeError

da = Vector3((1, 2, 3))
ma = Matrix3(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
