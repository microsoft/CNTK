package main

import "./director_unroll"

type MyFoo struct{} // From director_unroll.Foo
func (p *MyFoo) Ping() string {
	return "MyFoo::ping()"
}

func main() {
	a := director_unroll.NewDirectorFoo(&MyFoo{})

	b := director_unroll.NewBar()

	b.Set(a)
	c := b.Get()

	if c.Ping() != "MyFoo::ping()" {
		panic(c.Ping())
	}
}
