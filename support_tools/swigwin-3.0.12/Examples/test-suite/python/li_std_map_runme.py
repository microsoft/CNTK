import li_std_map

a1 = li_std_map.A(3)
a2 = li_std_map.A(7)


p0 = li_std_map.pairii(1, 2)
p1 = li_std_map.pairA(1, a1.this)
m = {}
m[1] = a1
m[2] = a2

pp1 = li_std_map.p_identa(p1)
mm = li_std_map.m_identa(m)


m = li_std_map.mapA()
m[1] = a1
m[2] = a2


pm = {}
for k in m:
    pm[k] = m[k]

for k in m:
    if pm[k].this != m[k].this:
        print pm[k], m[k]
        raise RuntimeError


m = {}
m[1] = (1, 2)
m["foo"] = "hello"

pm = li_std_map.pymap()

for k in m:
    pm[k] = m[k]

for k in pm:
    if (pm[k] != m[k]):
        raise RuntimeError


mii = li_std_map.IntIntMap()

mii[1] = 1
mii[1] = 2

if mii[1] != 2:
    raise RuntimeError
