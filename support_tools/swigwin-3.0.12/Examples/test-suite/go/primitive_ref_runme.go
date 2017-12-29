package main

import . "./primitive_ref"

func main() {
	if Ref_int(3) != 3 {
		panic(0)
	}

	if Ref_uint(3) != 3 {
		panic(0)
	}

	if Ref_short(3) != 3 {
		panic(0)
	}

	if Ref_ushort(3) != 3 {
		panic(0)
	}

	if Ref_long(3) != 3 {
		panic(0)
	}

	if Ref_ulong(3) != 3 {
		panic(0)
	}

	if Ref_schar(3) != 3 {
		panic(0)
	}

	if Ref_uchar(3) != 3 {
		panic(0)
	}

	if Ref_float(3.5) != 3.5 {
		panic(0)
	}

	if Ref_double(3.5) != 3.5 {
		panic(0)
	}

	if Ref_bool(true) != true {
		panic(0)
	}

	if Ref_char('x') != 'x' {
		panic(0)
	}

	if Ref_over(0) != 0 {
		panic(0)
	}
}
