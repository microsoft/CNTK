package main

import . "./director_finalizer"

type MyFoo struct{} // From Foo
func DeleteMyFoo(p Foo) {
	p.OrStatus(2)
	DeleteFoo(p)
}

func main() {
	ResetStatus()

	a := NewDirectorFoo(&MyFoo{})
	DeleteMyFoo(a)

	if GetStatus() != 3 {
		panic(0)
	}

	ResetStatus()

	a = NewDirectorFoo(&MyFoo{})
	Launder(a)

	if GetStatus() != 0 {
		panic(0)
	}

	DeleteMyFoo(a)

	if GetStatus() != 3 {
		panic(0)
	}

	ResetStatus()
}
