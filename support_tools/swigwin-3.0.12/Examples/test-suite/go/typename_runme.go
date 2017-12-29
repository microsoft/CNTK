package main

import "./typename"

func main() {
	f := typename.NewFoo()
	b := typename.NewBar()

	var x float64 = typename.TwoFoo(f)
	var y int = typename.TwoBar(b)
	_ = x
	_ = y
}
