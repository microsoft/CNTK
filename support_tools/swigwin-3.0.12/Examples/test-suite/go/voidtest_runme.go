package main

import "./voidtest"

func main() {
	voidtest.Globalfunc()
	f := voidtest.NewFoo()
	f.Memberfunc()

	voidtest.FooStaticmemberfunc()

	v1 := voidtest.Vfunc1(f.Swigcptr())
	v2 := voidtest.Vfunc2(f)
	if v1 != v2 {
		panic(0)
	}

	v3 := voidtest.Vfunc3(v1)

	v4 := voidtest.Vfunc1(f.Swigcptr())
	if v4 != v1 {
		panic(0)
	}

	v3.Memberfunc()
}
