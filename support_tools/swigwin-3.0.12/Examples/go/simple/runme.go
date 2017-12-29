package main

import (
	"./example"
	"fmt"
)

func main() {
	// Call our gcd() function
	x := 42
	y := 105
	g := example.Gcd(x, y)
	fmt.Println("The gcd of", x, "and", y, "is", g)

	// Manipulate the Foo global variable

	// Output its current value
	fmt.Println("Foo =", example.GetFoo())

	// Change its value
	example.SetFoo(3.1415926)

	// See if the change took effect
	fmt.Println("Foo =", example.GetFoo())
}
