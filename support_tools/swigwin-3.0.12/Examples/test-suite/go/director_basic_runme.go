package main

import "./director_basic"

type GoFoo struct{}

func (p *GoFoo) Ping() string {
	return "GoFoo::ping()"
}

func f1() {
	a := director_basic.NewDirectorFoo(&GoFoo{})

	if a.Ping() != "GoFoo::ping()" {
		panic(a.Ping())
	}

	if a.Pong() != "Foo::pong();GoFoo::ping()" {
		panic(a.Pong())
	}

	b := director_basic.NewFoo()

	if b.Ping() != "Foo::ping()" {
		panic(b.Ping())
	}

	if b.Pong() != "Foo::pong();Foo::ping()" {
		panic(b.Pong())
	}

	a1 := director_basic.NewA1(1)

	if a1.Rg(2) != 2 {
		panic(0)
	}
}

type GoClass struct {
	cmethod int
}

func (p *GoClass) Method(uintptr) {
	p.cmethod = 7
}
func (p *GoClass) Vmethod(b director_basic.Bar) director_basic.Bar {
	b.SetX(b.GetX() + 31)
	return b
}

var bc director_basic.Bar

func f2() {
	b := director_basic.NewBar(3)
	d := director_basic.NewMyClass()
	pc := &GoClass{0}
	c := director_basic.NewDirectorMyClass(pc)

	cc := director_basic.MyClassGet_self(c)
	dd := director_basic.MyClassGet_self(d)

	bc = cc.Cmethod(b)
	bd := dd.Cmethod(b)

	cc.Method(b.Swigcptr())
	if pc.cmethod != 7 {
		panic(pc.cmethod)
	}

	if bc.GetX() != 34 {
		panic(bc.GetX())
	}

	if bd.GetX() != 16 {
		panic(bd.GetX())
	}
}

type GoMulti struct {
	GoClass
}

func (p *GoMulti) Vmethod(b director_basic.Bar) director_basic.Bar {
	b.SetX(b.GetX() + 31)
	return b
}
func (p *GoMulti) Ping() string {
	return "GoFoo::ping()"
}

func f3() {
	for i := 0; i < 100; i++ {
		p := &GoMulti{GoClass{0}}
		gomult := director_basic.NewDirectorFoo(p)
		gomult.Pong()
		director_basic.DeleteDirectorFoo(gomult)
	}

	p := &GoMulti{GoClass{0}}
	gomult := director_basic.NewDirectorMyClass(p)
	fgomult := director_basic.NewDirectorFoo(gomult)

	p1 := director_basic.FooGet_self(fgomult.(director_basic.Foo))
	p2 := director_basic.MyClassGet_self(gomult.(director_basic.MyClass))

	p1.Ping()
	p2.Vmethod(bc)
}

func main() {
	f1()
	f2()
	f3()
}
