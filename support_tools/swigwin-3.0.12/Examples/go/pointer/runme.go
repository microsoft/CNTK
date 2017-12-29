package main

import (
	. "./example"
	"fmt"
)

func main() {
	// First create some objects using the pointer library.
	fmt.Println("Testing the pointer library")
	a := New_intp()
	b := New_intp()
	c := New_intp()
	Intp_assign(a, 37)
	Intp_assign(b, 42)

	fmt.Println("     a =", a)
	fmt.Println("     b =", b)
	fmt.Println("     c =", c)

	// Call the add() function with some pointers
	Add(a, b, c)

	// Now get the result
	res := Intp_value(c)
	fmt.Println("     37 + 42 =", res)

	// Clean up the pointers
	Delete_intp(a)
	Delete_intp(b)
	Delete_intp(c)

	// Now try the typemap library
	// Now it is no longer necessary to manufacture pointers.
	// Instead we use a single element slice which in Go is modifiable.

	fmt.Println("Trying the typemap library")
	r := []int{0}
	Sub(37, 42, r)
	fmt.Println("     37 - 42 = ", r[0])

	// Now try the version with return value

	fmt.Println("Testing return value")
	q := Divide(42, 37, r)
	fmt.Println("     42/37 = ", q, " remainder ", r[0])
}
