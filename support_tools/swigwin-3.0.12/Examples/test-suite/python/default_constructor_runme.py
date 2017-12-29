import _default_constructor

# This test is expected to fail with -builtin option.
# It uses the old static syntax (e.g., dc.new_A() rather than dc.A()),
# which is not provided with the -builtin option.
if _default_constructor.is_python_builtin():
    exit(0)

dc = _default_constructor

a = dc.new_A()
dc.delete_A(a)

aa = dc.new_AA()
dc.delete_AA(aa)

try:
    b = dc.new_B()
    print "Whoa. new_BB created."
except:
    pass

del_b = dc.delete_B

try:
    bb = dc.new_BB()
    print "Whoa. new_BB created."
except:
    pass

del_bb = dc.delete_BB

try:
    c = dc.new_C()
    print "Whoa. new_C created."
except:
    pass

del_c = dc.delete_C

cc = dc.new_CC()
dc.delete_CC(cc)

try:
    d = dc.new_D()
    print "Whoa. new_D created"
except:
    pass

del_d = dc.delete_D

try:
    dd = dc.new_DD()
    print "Whoa. new_DD created"
except:
    pass

dd = dc.delete_DD

try:
    ad = dc.new_AD()
    print "Whoa. new_AD created"
except:
    pass

del_ad = dc.delete_AD

e = dc.new_E()
dc.delete_E(e)

ee = dc.new_EE()
dc.delete_EE(ee)

try:
    eb = dc.new_EB()
    print "Whoa. new_EB created"
except:
    pass

del_eb = dc.delete_EB

f = dc.new_F()

try:
    del_f = dc.delete_F
    print "Whoa. delete_F created"
except AttributeError:
    pass

dc.F_destroy(f)

g = dc.new_G()

try:
    del_g = dc.delete_G
    print "Whoa. delete_G created"
except AttributeError:
    pass

dc.G_destroy(g)

gg = dc.new_GG()
dc.delete_GG(gg)


import default_constructor
hh = default_constructor.HH(1, 1)
