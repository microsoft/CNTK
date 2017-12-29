package main

import "./virtual_poly"

func main() {
	d := virtual_poly.NewNDouble(3.5)
	i := virtual_poly.NewNInt(2)

	// the copy methods return the right polymorphic types
	dc := d.Copy()
	ic := i.Copy()

	if d.Get() != dc.Get() {
		panic(0)
	}

	if i.Get() != ic.Get() {
		panic(0)
	}

	virtual_poly.Incr(ic)

	if (i.Get() + 1) != ic.Get() {
		panic(0)
	}

	dr := d.Ref_this()
	if d.Get() != dr.Get() {
		panic(0)
	}

	// 'narrowing' also works
	ddc := virtual_poly.NDoubleNarrow(d.Nnumber())
	if d.Get() != ddc.Get() {
		panic(0)
	}

	dic := virtual_poly.NIntNarrow(i.Nnumber())
	if i.Get() != dic.Get() {
		panic(0)
	}
}
