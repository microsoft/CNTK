package main

import "./typedef_class"

func main() {
	a := typedef_class.NewRealA()
	a.SetA(3)

	b := typedef_class.NewB()
	b.TestA(a)
}
