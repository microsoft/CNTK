import string
from template_typedef_cplx4 import *

#
# this is OK
#


s = Sin()
s.get_base_value()
s.get_value()
s.get_arith_value()
my_func_r(s)
make_Multiplies_double_double_double_double(s, s)

z = CSin()
z.get_base_value()
z.get_value()
z.get_arith_value()
my_func_c(z)
make_Multiplies_complex_complex_complex_complex(z, z)

#
# Here we fail
#
d = make_Identity_double()
my_func_r(d)

c = make_Identity_complex()
my_func_c(c)
