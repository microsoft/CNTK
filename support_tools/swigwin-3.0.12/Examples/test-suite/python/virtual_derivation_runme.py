from virtual_derivation import *
#
# very innocent example
#
b = B(3)
if b.get_a() != b.get_b():
    raise RuntimeError, "something is really wrong"
