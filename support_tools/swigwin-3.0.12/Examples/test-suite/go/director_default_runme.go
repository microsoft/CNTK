package main

import . "./director_default"

func main() {
	NewFoo()
	NewFoo(1)

	NewBar()
	NewBar(1)
}
