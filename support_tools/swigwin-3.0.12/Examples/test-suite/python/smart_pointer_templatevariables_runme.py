from smart_pointer_templatevariables import *

d = DiffImContainerPtr_D(create(1234, 5678))

if (d.id != 1234):
    raise RuntimeError
# if (d.xyz != 5678):
#  raise RuntimeError

d.id = 4321
#d.xyz = 8765

if (d.id != 4321):
    raise RuntimeError
# if (d.xyz != 8765):
#  raise RuntimeError
