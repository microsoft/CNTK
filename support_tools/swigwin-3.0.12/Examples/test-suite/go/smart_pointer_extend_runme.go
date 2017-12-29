package main

import . "./smart_pointer_extend"

func main() {
	f := NewFoo()
	b := NewBar(f)

	if b.Extension() != f.Extension() {
		panic(0)
	}

	b2 := NewCBase()
	d := NewCDerived()
	p := NewCPtr()

	if b2.Bar() != p.Bar() {
		panic(0)
	}

	if d.Foo() != p.Foo() {
		panic(0)
	}

	if CBaseHello() != p.Hello() {
		panic(0)
	}

	d2 := NewDFoo()

	dp := NewDPtrFoo(d2)

	if DFooSExt(1) != dp.SExt(1) {
		panic(0)
	}

	if d2.Ext(1) != dp.Ext(1) {
		panic(0)
	}
}
