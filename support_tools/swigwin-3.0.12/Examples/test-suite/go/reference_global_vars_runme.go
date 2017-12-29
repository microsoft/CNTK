package main

import . "./reference_global_vars"

func main() {
	// const class reference variable
	if GetconstTC().GetNum() != 33 {
		panic(0)
	}

	// primitive reference variables
	SetVar_bool(Createref_bool(false))
	if Value_bool(GetVar_bool()) != false {
		println(1, GetVar_bool(), Value_bool(GetVar_bool()))
		panic(0)
	}

	SetVar_bool(Createref_bool(true))
	if Value_bool(GetVar_bool()) != true {
		println(2, GetVar_bool(), Value_bool(GetVar_bool()))
		panic(0)
	}

	SetVar_char(Createref_char('w'))
	if Value_char(GetVar_char()) != 'w' {
		println(3, GetVar_char(), Value_char(GetVar_char()))
		panic(0)
	}

	SetVar_unsigned_char(Createref_unsigned_char(10))
	if Value_unsigned_char(GetVar_unsigned_char()) != 10 {
		println(4, GetVar_unsigned_char(), Value_unsigned_char(GetVar_unsigned_char()))
		panic(0)
	}

	SetVar_signed_char(Createref_signed_char(10))
	if Value_signed_char(GetVar_signed_char()) != 10 {
		panic(0)
	}

	SetVar_short(Createref_short(10))
	if Value_short(GetVar_short()) != 10 {
		panic(0)
	}

	SetVar_unsigned_short(Createref_unsigned_short(10))
	if Value_unsigned_short(GetVar_unsigned_short()) != 10 {
		panic(0)
	}

	SetVar_int(Createref_int(10))
	if Value_int(GetVar_int()) != 10 {
		panic(0)
	}

	SetVar_unsigned_int(Createref_unsigned_int(10))
	if Value_unsigned_int(GetVar_unsigned_int()) != 10 {
		panic(0)
	}

	SetVar_long(Createref_long(10))
	if Value_long(GetVar_long()) != 10 {
		panic(0)
	}

	SetVar_unsigned_long(Createref_unsigned_long(10))
	if Value_unsigned_long(GetVar_unsigned_long()) != 10 {
		panic(0)
	}

	SetVar_long_long(Createref_long_long(0x6FFFFFFFFFFFFFF8))
	if Value_long_long(GetVar_long_long()) != 0x6FFFFFFFFFFFFFF8 {
		panic(0)
	}

	//ull = abs(0xFFFFFFF2FFFFFFF0)
	ull := uint64(55834574864)
	SetVar_unsigned_long_long(Createref_unsigned_long_long(ull))
	if Value_unsigned_long_long(GetVar_unsigned_long_long()) != ull {
		panic(0)
	}

	SetVar_float(Createref_float(10.5))
	if Value_float(GetVar_float()) != 10.5 {
		panic(0)
	}

	SetVar_double(Createref_double(10.5))
	if Value_double(GetVar_double()) != 10.5 {
		panic(0)
	}

	// class reference variable
	SetVar_TestClass(Createref_TestClass(NewTestClass(20)))
	if Value_TestClass(GetVar_TestClass()).GetNum() != 20 {
		panic(0)
	}
}
