from default_arg_values import *

d = Display()

if d.draw1() != 0:
    raise RuntimeError

if d.draw1(12) != 12:
    raise RuntimeError

p = createPtr(123)
if d.draw2() != 0:
    raise RuntimeError

if d.draw2(p) != 123:
    raise RuntimeError

if d.bool0() != False or type(d.bool0()) != type(False):
    raise RuntimeError

if d.bool1() != True or type(d.bool1()) != type(True):
    raise RuntimeError

if d.mybool0() != False or type(d.mybool0()) != type(False):
    raise RuntimeError

if d.mybool1() != True or type(d.mybool1()) != type(True):
    raise RuntimeError
