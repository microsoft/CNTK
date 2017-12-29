package main

import . "./struct_initialization"

func main() {
	if GetInstanceC1().GetX() != 10 {
		panic(0)
	}

	if GetInstanceD1().GetX() != 10 {
		panic(0)
	}

	if GetInstanceD2().GetX() != 20 {
		panic(0)
	}

	if GetInstanceD3().GetX() != 30 {
		panic(0)
	}

	if GetInstanceE1().GetX() != 1 {
		panic(0)
	}

	if GetInstanceF1().GetX() != 1 {
		panic(0)
	}
}
