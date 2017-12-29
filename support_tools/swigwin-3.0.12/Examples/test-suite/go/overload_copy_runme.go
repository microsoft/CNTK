package main

import . "./overload_copy"

func main() {
	f := NewFoo()
	_ = NewFoo(f)
}
