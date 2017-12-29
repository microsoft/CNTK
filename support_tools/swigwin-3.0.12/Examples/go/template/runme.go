// This example illustrates how C++ templates can be used from Go.

package main

import (
	. "./example"
	"fmt"
)

func main() {

	// Call some templated functions
	fmt.Println(Maxint(3, 7))
	fmt.Println(Maxdouble(3.14, 2.18))

	// Create some class
	iv := NewVecint(100)
	dv := NewVecdouble(1000)

	for i := 0; i < 100; i++ {
		iv.Setitem(i, 2*i)
	}

	for i := 0; i < 1000; i++ {
		dv.Setitem(i, 1.0/float64(i+1))
	}

	{
		sum := 0
		for i := 0; i < 100; i++ {
			sum = sum + iv.Getitem(i)
		}
		fmt.Println(sum)
	}

	{
		sum := float64(0.0)
		for i := 0; i < 1000; i++ {
			sum = sum + dv.Getitem(i)
		}
		fmt.Println(sum)
	}
}
