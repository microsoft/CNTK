package main

import . "./template_typedef_cplx4"

func main() {
	// this is OK

	s := NewSin()
	s.Get_base_value()
	s.Get_value()
	s.Get_arith_value()
	My_func_r(s)
	Make_Multiplies_double_double_double_double(s, s)

	z := NewCSin()
	z.Get_base_value()
	z.Get_value()
	z.Get_arith_value()
	My_func_c(z)
	Make_Multiplies_complex_complex_complex_complex(z, z)

	// Here we fail
	d := Make_Identity_double()
	My_func_r(d)

	c := Make_Identity_complex()
	My_func_c(c)
}
