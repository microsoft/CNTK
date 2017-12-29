package main

import "./grouping"

func main() {
	x := grouping.Test1(42)
	if x != 42 {
		panic(0)
	}

	grouping.Test2(42)

	x = grouping.Do_unary(37, grouping.NEGATE)
	if x != -37 {
		panic(0)
	}

	grouping.SetTest3(42)
}
