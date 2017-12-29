package main

import . "./template_inherit"

func main() {
	a := NewFooInt()
	b := NewFooDouble()
	c := NewBarInt()
	d := NewBarDouble()
	e := NewFooUInt()
	f := NewBarUInt()

	if a.Blah() != "Foo" {
		panic(0)
	}

	if b.Blah() != "Foo" {
		panic(0)
	}

	if e.Blah() != "Foo" {
		panic(0)
	}

	if c.Blah() != "Bar" {
		panic(0)
	}

	if d.Blah() != "Bar" {
		panic(0)
	}

	if f.Blah() != "Bar" {
		panic(0)
	}

	if c.Foomethod() != "foomethod" {
		panic(0)
	}

	if d.Foomethod() != "foomethod" {
		panic(0)
	}

	if f.Foomethod() != "foomethod" {
		panic(0)
	}

	if Invoke_blah_int(a) != "Foo" {
		panic(0)
	}

	if Invoke_blah_int(c) != "Bar" {
		panic(0)
	}

	if Invoke_blah_double(b) != "Foo" {
		panic(0)
	}

	if Invoke_blah_double(d) != "Bar" {
		panic(0)
	}

	if Invoke_blah_uint(e) != "Foo" {
		panic(0)
	}

	if Invoke_blah_uint(f) != "Bar" {
		panic(0)
	}
}
