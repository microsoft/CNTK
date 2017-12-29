package main

import "fmt"
import . "./smart_pointer_member"

func main() {
	f := NewFoo()
	f.SetY(1)

	if f.GetY() != 1 {
		panic(0)
	}

	b := NewBar(f)
	b.SetY(2)

	if f.GetY() != 2 {
		fmt.Println(f.GetY())
		fmt.Println(b.GetY())
		panic(0)
	}

	if b.GetX() != f.GetX() {
		panic(0)
	}

	if b.GetZ() != GetFooZ() {
		panic(0)
	}
}
