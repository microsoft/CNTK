from array_member import *

f = Foo()
f.data = cvar.global_data

for i in range(0, 8):
    if get_value(f.data, i) != get_value(cvar.global_data, i):
        raise RuntimeError, "Bad array assignment"


for i in range(0, 8):
    set_value(f.data, i, -i)

cvar.global_data = f.data

for i in range(0, 8):
    if get_value(f.data, i) != get_value(cvar.global_data, i):
        raise RuntimeError, "Bad array assignment"
