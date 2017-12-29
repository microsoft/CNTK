from cpp11_type_aliasing import *

if get_host_target().bits != 32:
    raise RuntimeError("get_host_target().bits should return 32")

if mult2(10) != 20:
    raise RuntimeError("mult2(10) should return 20")

int_ptr = allocate_int()
inplace_mult2(int_ptr)
if read_int(int_ptr) != 24:
    raise RuntimeError("read_int should return 24")
free_int(int_ptr)

pair = PairSubclass(3, 4)
if pair.first() != 3:
    raise RuntimeError("pair.first() should return 3")

if pair.second() != 4:
    raise RuntimeError("pair.second() should return 4")

if pair.a != 3:
    raise RuntimeError("pair.a should be 3")

if plus1(5) != 6:
    raise RuntimeError("plus1(5) should return 6")

if call(mult2_cb, 7) != 14:
    raise RuntimeError("call(mult2_cb, 7) should return 14")

if call(get_callback(), 7) != 14:
    raise RuntimeError("call(get_callback(), 7) should return 14")
