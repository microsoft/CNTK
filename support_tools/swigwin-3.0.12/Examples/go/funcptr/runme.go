package main

import (
	. "./example"
	"fmt"
)

func main() {
	a := 37
	b := 42

	// Now call our C function with a bunch of callbacks

	fmt.Println("Trying some C callback functions")
	fmt.Println("    a        = ", a)
	fmt.Println("    b        = ", b)
	fmt.Println("    ADD(a,b) = ", Do_op(a, b, ADD))
	fmt.Println("    SUB(a,b) = ", Do_op(a, b, SUB))
	fmt.Println("    MUL(a,b) = ", Do_op(a, b, MUL))

	fmt.Println("Here is what the C callback function classes are called in Go")
	fmt.Println("    ADD      = ", ADD)
	fmt.Println("    SUB      = ", SUB)
	fmt.Println("    MUL      = ", MUL)
}
