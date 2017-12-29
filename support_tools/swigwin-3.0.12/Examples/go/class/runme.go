// This example illustrates how C++ classes can be used from Go using SWIG.

package main

import (
	. "./example"
	"fmt"
)

func main() {
	// ----- Object creation -----

	fmt.Println("Creating some objects:")
	c := NewCircle(10)
	fmt.Println("   Created circle", c)
	s := NewSquare(10)
	fmt.Println("   Created square", s)

	// ----- Access a static member -----

	fmt.Println("\nA total of", GetShapeNshapes(), "shapes were created")

	// ----- Member data access -----

	// Notice how we can do this using functions specific to
	// the 'Circle' class.
	c.SetX(20)
	c.SetY(30)

	// Now use the same functions in the base class
	var shape Shape = s
	shape.SetX(-10)
	shape.SetY(5)

	fmt.Println("\nHere is their current position:")
	fmt.Println("    Circle = (", c.GetX(), " ", c.GetY(), ")")
	fmt.Println("    Square = (", s.GetX(), " ", s.GetY(), ")")

	// ----- Call some methods -----

	fmt.Println("\nHere are some properties of the shapes:")
	shapes := []Shape{c, s}
	for i := 0; i < len(shapes); i++ {
		fmt.Println("   ", shapes[i])
		fmt.Println("        area      = ", shapes[i].Area())
		fmt.Println("        perimeter = ", shapes[i].Perimeter())
	}

	// Notice how the area() and perimeter() functions really
	// invoke the appropriate virtual method on each object.

	// ----- Delete everything -----

	fmt.Println("\nGuess I'll clean up now")

	// Note: this invokes the virtual destructor
	// You could leave this to the garbage collector
	DeleteCircle(c)
	DeleteSquare(s)

	fmt.Println(GetShapeNshapes(), " shapes remain")
	fmt.Println("Goodbye")
}
