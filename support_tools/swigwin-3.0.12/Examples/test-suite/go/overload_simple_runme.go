package main

import . "./overload_simple"

func main() {
	if Foo(3) != "foo:int" {
		panic("foo(int)")
	}

	if Foo(3.0) != "foo:double" {
		panic("foo(double)")
	}

	if Foo("hello") != "foo:char *" {
		panic("foo(char *)")
	}

	f := NewFoos()
	b := NewBar()

	if Foo(f) != "foo:Foo *" {
		panic("foo(Foo *)")
	}

	if Foo(b) != "foo:Bar *" {
		panic("foo(Bar *)")
	}

	v := Malloc_void(32)

	if Foo(v) != "foo:void *" {
		panic("foo(void *)")
	}
	s := NewSpam()

	if s.Foo(3) != "foo:int" {
		panic("Spam::foo(int)")
	}

	if s.Foo(3.0) != "foo:double" {
		panic("Spam::foo(double)")
	}

	if s.Foo("hello") != "foo:char *" {
		panic("Spam::foo(char *)")
	}

	if s.Foo(f) != "foo:Foo *" {
		panic("Spam::foo(Foo *)")
	}

	if s.Foo(b) != "foo:Bar *" {
		panic("Spam::foo(Bar *)")
	}

	if s.Foo(v) != "foo:void *" {
		panic("Spam::foo(void *)")
	}

	if SpamBar(3) != "bar:int" {
		panic("Spam::bar(int)")
	}

	if SpamBar(3.0) != "bar:double" {
		panic("Spam::bar(double)")
	}

	if SpamBar("hello") != "bar:char *" {
		panic("Spam::bar(char *)")
	}

	if SpamBar(f) != "bar:Foo *" {
		panic("Spam::bar(Foo *)")
	}

	if SpamBar(b) != "bar:Bar *" {
		panic("Spam::bar(Bar *)")
	}

	if SpamBar(v) != "bar:void *" {
		panic("Spam::bar(void *)")
	}

	// Test constructors

	s = NewSpam()
	if s.GetXtype() != "none" {
		panic("Spam()")
	}

	s = NewSpam(3)
	if s.GetXtype() != "int" {
		panic("Spam(int)")
	}

	s = NewSpam(3.4)
	if s.GetXtype() != "double" {
		panic("Spam(double)")
	}

	s = NewSpam("hello")
	if s.GetXtype() != "char *" {
		panic("Spam(char *)")
	}

	s = NewSpam(f)
	if s.GetXtype() != "Foo *" {
		panic("Spam(Foo *)")
	}

	s = NewSpam(b)
	if s.GetXtype() != "Bar *" {
		panic("Spam(Bar *)")
	}

	s = NewSpam(v)
	if s.GetXtype() != "void *" {
		panic("Spam(void *)")
	}

	Free_void(v)

	a := NewClassA()
	_ = a.Method1(1)
}
