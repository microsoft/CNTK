from template_ns import *
p1 = pairii(2, 3)
p2 = pairii(p1)

if p2.first != 2:
    raise RuntimeError
if p2.second != 3:
    raise RuntimeError

p3 = pairdd(3.5, 2.5)
p4 = pairdd(p3)

if p4.first != 3.5:
    raise RuntimeError

if p4.second != 2.5:
    raise RuntimeError
