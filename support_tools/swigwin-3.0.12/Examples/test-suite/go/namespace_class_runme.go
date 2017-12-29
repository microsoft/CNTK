package main

import . "./namespace_class"

func main() {
	EulerT3DToFrame(1, 1, 1)

	_ = NewBooT_i()
	_ = NewBooT_H()

	f1 := NewFooT_i()
	f1.Quack(1)

	f2 := NewFooT_d()
	f2.Moo(1)

	f3 := NewFooT_H()
	f3.Foo(Hi)
}
