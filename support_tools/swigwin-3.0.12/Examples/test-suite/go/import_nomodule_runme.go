package main

import . "./import_nomodule"

func main() {
	f := Create_Foo()
	Test1(f, 42)
	Delete_Foo(f)

	b := NewBar()
	Test1(b, 37)
}
