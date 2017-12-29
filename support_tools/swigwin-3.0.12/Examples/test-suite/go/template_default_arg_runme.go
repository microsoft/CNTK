package main

import "./template_default_arg"

func main() {
	helloInt := template_default_arg.NewHello_int()
	helloInt.Foo(template_default_arg.Hello_intHi)

	x := template_default_arg.NewX_int()
	if x.Meth(20.0, 200).(int) != 200 {
		panic("X_int test 1 failed")
	}
	if x.Meth(20).(int) != 20 {
		panic("X_int test 2 failed")
	}
	if x.Meth().(int) != 0 {
		panic("X_int test 3 failed")
	}

	y := template_default_arg.NewY_unsigned()
	if y.Meth(20.0, uint(200)).(uint) != 200 {
		panic("Y_unsigned test 1 failed")
	}
	if y.Meth(uint(20)).(uint) != 20 {
		panic("Y_unsigned test 2 failed")
	}
	if y.Meth().(uint) != 0 {
		panic("Y_unsigned test 3 failed")
	}

	_ = template_default_arg.NewX_longlong()
	_ = template_default_arg.NewX_longlong(20.0)
	_ = template_default_arg.NewX_longlong(20.0, int64(200))

	_ = template_default_arg.NewX_int()
	_ = template_default_arg.NewX_int(20.0)
	_ = template_default_arg.NewX_int(20.0, 200)

	_ = template_default_arg.NewX_hello_unsigned()
	_ = template_default_arg.NewX_hello_unsigned(20.0)
	_ = template_default_arg.NewX_hello_unsigned(20.0, template_default_arg.NewHello_int())

	yy := template_default_arg.NewY_hello_unsigned()
	yy.Meth(20.0, template_default_arg.NewHello_int())
	yy.Meth(template_default_arg.NewHello_int())
	yy.Meth()

	fz := template_default_arg.NewFoo_Z_8()
	xz := template_default_arg.NewX_Foo_Z_8()
	_ = xz.Meth(fz)

	// Templated functions

	// plain function: int ott(Foo<int>)
	if template_default_arg.Ott(template_default_arg.NewFoo_int()) != 30 {
		panic("ott test 1 failed")
	}

	// %template(ott) ott<int, int>
	if template_default_arg.Ott() != 10 {
		panic("ott test 2 failed")
	}
	if template_default_arg.Ott(1) != 10 {
		panic("ott test 3 failed")
	}
	if template_default_arg.Ott(1, 1) != 10 {
		panic("ott test 4 failed")
	}

	if template_default_arg.Ott("hi") != 20 {
		panic("ott test 5 failed")
	}
	if template_default_arg.Ott("hi", 1) != 20 {
		panic("ott test 6 failed")
	}
	if template_default_arg.Ott("hi", 1, 1) != 20 {
		panic("ott test 7 failed")
	}

	// %template(ott) ott<const char *>
	if template_default_arg.Ottstring(template_default_arg.NewHello_int(), "hi") != 40 {
		panic("ott test 8 failed")
	}

	if template_default_arg.Ottstring(template_default_arg.NewHello_int()) != 40 {
		panic("ott test 9 failed")
	}

	// %template(ott) ott<int>
	if template_default_arg.Ottint(template_default_arg.NewHello_int(), 1) != 50 {
		panic("ott test 10 failed")
	}

	if template_default_arg.Ottint(template_default_arg.NewHello_int()) != 50 {
		panic("ott test 11 failed")
	}

	// %template(ott) ott<double>
	if template_default_arg.Ott(template_default_arg.NewHello_int(), 1.0) != 60 {
		panic("ott test 12 failed")
	}

	if template_default_arg.Ott(template_default_arg.NewHello_int()) != 60 {
		panic("ott test 13 failed")
	}
}
