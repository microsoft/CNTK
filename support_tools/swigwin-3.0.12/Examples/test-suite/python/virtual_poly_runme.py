import virtual_poly

d = virtual_poly.NDouble(3.5)
i = virtual_poly.NInt(2)

#
# the copy methods return the right polymorphic types
#
dc = d.copy()
ic = i.copy()

if d.get() != dc.get():
    raise RuntimeError

if i.get() != ic.get():
    raise RuntimeError

virtual_poly.incr(ic)

if (i.get() + 1) != ic.get():
    raise RuntimeError


dr = d.ref_this()
if d.get() != dr.get():
    raise RuntimeError


#
# 'narrowing' also works
#
ddc = virtual_poly.NDouble_narrow(d.nnumber())
if d.get() != ddc.get():
    raise RuntimeError

dic = virtual_poly.NInt_narrow(i.nnumber())
if i.get() != dic.get():
    raise RuntimeError
