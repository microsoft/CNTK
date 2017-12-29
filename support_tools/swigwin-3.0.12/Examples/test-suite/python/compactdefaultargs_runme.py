from compactdefaultargs import *

defaults1 = Defaults1(1000)
defaults1 = Defaults1()

if defaults1.ret(10.0) != 10.0:
    raise RuntimeError

if defaults1.ret() != -1.0:
    raise RuntimeError

defaults2 = Defaults2(1000)
defaults2 = Defaults2()

if defaults2.ret(10.0) != 10.0:
    raise RuntimeError

if defaults2.ret() != -1.0:
    raise RuntimeError

if defaults2.nodefault(-2) != -2:
    raise RuntimeError
