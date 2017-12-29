require("import")	-- the import fn
import("primitive_ref")	-- import code
pr=primitive_ref --alias

assert(pr.ref_int(3)==3)

assert(pr.ref_uint(3) == 3)

assert(pr.ref_short(3) == 3)

assert(pr.ref_ushort(3) == 3)

assert(pr.ref_long(3) == 3)

assert(pr.ref_ulong(3) == 3)

assert(pr.ref_schar(3) == 3)

assert(pr.ref_uchar(3) == 3)

assert(pr.ref_float(3.5) == 3.5)

assert(pr.ref_double(3.5) == 3.5)

assert(pr.ref_bool(true) == true)

assert(pr.ref_char('x') == 'x')

assert(pr.ref_over(0) == 0)

a=pr.A(12)
assert(pr.ref_over(a)==12)
