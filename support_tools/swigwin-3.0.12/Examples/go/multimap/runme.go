package main

import (
	. "./example"
	"fmt"
)

func main() {
	// Call our gcd() function
	x := 42
	y := 105
	g := Gcd(x, y)
	fmt.Println("The gcd of ", x, " and ", y, " is ", g)

	// Call the gcdmain() function
	args := []string{"gcdmain", "42", "105"}
	Gcdmain(args)

	// Call the count function
	fmt.Println(Count("Hello World", 'l'))

	// Call the capitalize function
	capitalizeMe := []string{"hello world"}
	Capitalize(capitalizeMe)
	fmt.Println(capitalizeMe[0])
}
