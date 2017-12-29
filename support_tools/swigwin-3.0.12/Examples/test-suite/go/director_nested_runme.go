package main

import . "./director_nested"

type A struct{} // From FooBar_int
func (p *A) Do_step() string {
	return "A::do_step;"
}
func (p *A) Get_value() string {
	return "A::get_value"
}

func f1() {
	a := NewDirectorFooBar_int(&A{})
	if a.Step() != "Bar::step;Foo::advance;Bar::do_advance;A::do_step;" {
		panic("Bad A virtual resolution")
	}
}

type B struct{} // From FooBar_int
func (p *B) Do_advance() string {
	return "B::do_advance;" + p.Do_step()
}
func (p *B) Do_step() string {
	return "B::do_step;"
}
func (p *B) Get_value() int {
	return 1
}

func f2() {
	b := NewDirectorFooBar_int(&B{})

	if b.Step() != "Bar::step;Foo::advance;B::do_advance;B::do_step;" {
		panic("Bad B virtual resolution")
	}
}

type C struct {
	fbi FooBar_int
} // From FooBar_int

func (p *C) Do_advance() string {
	return "C::do_advance;" + DirectorFooBar_intDo_advance(p.fbi)
}

func (p *C) Do_step() string {
	return "C::do_step;"
}

func (p *C) Get_value() int {
	return 2
}

func (p *C) Get_name() string {
	return DirectorFooBar_intGet_name(p.fbi) + " hello"
}

func f3() {
	m := &C{nil}
	cc := NewDirectorFooBar_int(m)
	m.fbi = cc
	c := FooBar_intGet_self(cc)
	c.Advance()

	if c.Get_name() != "FooBar::get_name hello" {
		panic(0)
	}

	if c.Name() != "FooBar::get_name hello" {
		panic(0)
	}
}

func main() {
	f1()
	f2()
	f3()
}
