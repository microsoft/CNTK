package main

import "./struct_value"

func main() {
	b := struct_value.NewBar()

	b.GetA().SetX(3)
	if b.GetA().GetX() != 3 {
		panic(0)
	}

	b.GetB().SetX(3)
	if b.GetB().GetX() != 3 {
		panic(0)
	}
}
