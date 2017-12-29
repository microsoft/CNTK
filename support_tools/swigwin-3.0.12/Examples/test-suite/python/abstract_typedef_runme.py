from abstract_typedef import *
e = Engine()

a = A()


if a.write(e) != 1:
    raise RuntimeError
