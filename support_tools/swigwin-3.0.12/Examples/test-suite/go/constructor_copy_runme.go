package main

import . "./constructor_copy"

func main() {
	f1 := NewFoo1(3)
	f11 := NewFoo1(f1)

	if f1.GetX() != f11.GetX() {
		panic("f1/f11 x mismatch")
	}

	bi := NewBari(5)
	bc := NewBari(bi)

	if bi.GetX() != bc.GetX() {
		panic("bi/bc x mismatch")
	}

	bd := NewBard(5)
	good := false
	func() {
		defer func() {
			if recover() != nil {
				good = true
			}
		}()
		NewBard(bd)
	}()

	if !good {
		panic("bd !good")
	}
}
