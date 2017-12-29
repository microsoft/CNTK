package main

import "./enums"

func main() {
	enums.Bar2(1)
	enums.Bar3(1)
	enums.Bar1(1)

	if enums.GetEnumInstance() != 2 {
		panic(0)
	}

	if enums.GetSlap() != 10 {
		panic(0)
	}

	if enums.GetMine() != 11 {
		panic(0)
	}

	if enums.GetThigh() != 12 {
		panic(0)
	}
}
