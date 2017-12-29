from li_std_vector import *

if typedef_test(101) != 101:
    raise RuntimeError

try:
  sv = StructVector([None, None])
  raise RuntimeError("Using None should result in a TypeError")
except TypeError:
  pass
