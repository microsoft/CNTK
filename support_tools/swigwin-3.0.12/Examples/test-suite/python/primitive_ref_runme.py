from primitive_ref import *

if ref_int(3) != 3:
    raise RuntimeError

if ref_uint(3) != 3:
    raise RuntimeError

if ref_short(3) != 3:
    raise RuntimeError

if ref_ushort(3) != 3:
    raise RuntimeError

if ref_long(3) != 3:
    raise RuntimeError

if ref_ulong(3) != 3:
    raise RuntimeError

if ref_schar(3) != 3:
    raise RuntimeError

if ref_uchar(3) != 3:
    raise RuntimeError

if ref_float(3.5) != 3.5:
    raise RuntimeError

if ref_double(3.5) != 3.5:
    raise RuntimeError

if ref_bool(True) != True:
    raise RuntimeError

if ref_char('x') != 'x':
    raise RuntimeError

if ref_over(0) != 0:
    raise RuntimeError
