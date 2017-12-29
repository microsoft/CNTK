import li_std_pair_extra

p = (1, 2)
p1 = li_std_pair_extra.p_inout(p)
p2 = li_std_pair_extra.p_inoutd(p1)

d1 = li_std_pair_extra.d_inout(2)

i, d2 = li_std_pair_extra.d_inout2(2)

i, p = li_std_pair_extra.p_inout2(p)
p3, p4 = li_std_pair_extra.p_inout3(p1, p1)

psi = li_std_pair_extra.SIPair("hello", 1)
pci = li_std_pair_extra.CIPair(1, 1)


#psi.first = "hi"


psi = li_std_pair_extra.SIPair("hi", 1)
if psi != ("hi", 1):
    raise RuntimeError

psii = li_std_pair_extra.SIIPair(psi, 1)

a = li_std_pair_extra.A()
b = li_std_pair_extra.B()

pab = li_std_pair_extra.ABPair(a, b)

pab.first = a
pab.first.val = 2

if pab.first.val != 2:
    raise RuntimeError


pci = li_std_pair_extra.CIntPair(1, 0)

a = li_std_pair_extra.A(5)
p1 = li_std_pair_extra.pairP1(1, a.this)
p2 = li_std_pair_extra.pairP2(a, 1)
p3 = li_std_pair_extra.pairP3(a, a)


if a.val != li_std_pair_extra.p_identa(p1.this)[1].val:
    raise RuntimeError

p = li_std_pair_extra.IntPair(1, 10)
p.first = 1

p = li_std_pair_extra.paircA1(1, a)
p.first
p.second

p = li_std_pair_extra.paircA2(1, a)
pp = li_std_pair_extra.pairiiA(1, p)
