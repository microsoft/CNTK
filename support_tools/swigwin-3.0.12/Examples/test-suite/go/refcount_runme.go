package main

import . "./refcount"

// very innocent example

func main() {
	a := NewA3()
	_ = NewB(a)
	b2 := BCreate(a)

	if a.Ref_count() != 3 {
		panic("This program will crash... now")
	}

	rca := b2.Get_rca()
	// _ = BCreate(rca)
	_ = rca

	if a.Ref_count() != 4 {
		panic("This program will crash... now")
	}

	/* Requires smart pointer support.
	v := NewVector_A(2)
	v.Set(0, a)
	v.Set(1, a)

	_ = v.Get(0)
	DeleteVector_A(v)
	*/
}
