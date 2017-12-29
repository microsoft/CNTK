from cpp11_type_traits import *

if Elaborate(0, 0) != 1:
    raise RuntimeError("Elaborate should have returned 1")

if Elaborate(0, 0.0) != 2:
    raise RuntimeError("Elaborate should have returned 2")
