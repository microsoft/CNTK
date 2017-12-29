from global_namespace import *


def is_new_style_class(cls):
    return hasattr(cls, "__class__")

k1 = Klass1()
k2 = Klass2()
k3 = Klass3()
k4 = Klass4()
k5 = Klass5()
k6 = Klass6()
k7 = Klass7()

if is_new_style_class(KlassMethods):
    KlassMethods_static = KlassMethods
else:
    KlassMethods_static = KlassMethods()
KlassMethods_static.methodA(k1, k2, k3, k4, k5, k6, k7)
KlassMethods_static.methodB(k1, k2, k3, k4, k5, k6, k7)

k1 = getKlass1A()
k2 = getKlass2A()
k3 = getKlass3A()
k4 = getKlass4A()
k5 = getKlass5A()
k6 = getKlass6A()
k7 = getKlass7A()

KlassMethods_static.methodA(k1, k2, k3, k4, k5, k6, k7)
KlassMethods_static.methodB(k1, k2, k3, k4, k5, k6, k7)

k1 = getKlass1B()
k2 = getKlass2B()
k3 = getKlass3B()
k4 = getKlass4B()
k5 = getKlass5B()
k6 = getKlass6B()
k7 = getKlass7B()

KlassMethods_static.methodA(k1, k2, k3, k4, k5, k6, k7)
KlassMethods_static.methodB(k1, k2, k3, k4, k5, k6, k7)

if is_new_style_class(XYZMethods):
    XYZMethods_static = XYZMethods
else:
    XYZMethods_static = XYZMethods()
XYZMethods_static.methodA(
    XYZ1(), XYZ2(), XYZ3(), XYZ4(), XYZ5(), XYZ6(), XYZ7())
XYZMethods_static.methodB(
    XYZ1(), XYZ2(), XYZ3(), XYZ4(), XYZ5(), XYZ6(), XYZ7())

if is_new_style_class(TheEnumMethods):
    TheEnumMethods_static = TheEnumMethods
else:
    TheEnumMethods_static = TheEnumMethods()
TheEnumMethods_static.methodA(theenum1, theenum2, theenum3)
TheEnumMethods_static.methodA(theenum1, theenum2, theenum3)
