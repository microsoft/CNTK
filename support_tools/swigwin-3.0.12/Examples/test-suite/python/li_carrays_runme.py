from li_carrays import *

d = doubleArray(10)

d[0] = 7
d[5] = d[0] + 3

if d[5] + d[0] != 17:
    raise RuntimeError

shorts = shortArray(5)

sum = sum_array(shorts)
if sum != 0:
    raise RuntimeError("incorrect zero sum, got: " + str(sum))

for i in range(5):
    shorts[i] = i

sum = sum_array(shorts)
if sum != 0+1+2+3+4:
    raise RuntimeError("incorrect sum, got: " + str(sum))
