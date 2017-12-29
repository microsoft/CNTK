package main

import . "./nested_workaround"

func main() {
	inner := NewInner(5)
	outer := NewOuter()
	newInner := outer.DoubleInnerValue(inner)
	if newInner.GetValue() != 10 {
		panic(0)
	}

	outer = NewOuter()
	inner = outer.CreateInner(3)
	newInner = outer.DoubleInnerValue(inner)
	if outer.GetInnerValue(newInner) != 6 {
		panic(0)
	}
}
