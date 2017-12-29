package main

import . "./director_exception"

type Exception struct {
	msg string
}

func NewException(a, b string) *Exception {
	return &Exception{a + b}
}

type MyFoo struct{} // From Foo
func (p *MyFoo) Ping() string {
	panic("MyFoo::ping() EXCEPTION")
}

type MyFoo2 struct{} // From Foo
func (p *MyFoo2) Ping() bool {
	return true // should return a string
}

type MyFoo3 struct{} // From Foo
func (p *MyFoo3) Ping() string {
	panic(NewException("foo", "bar"))
}

func main() {
	// Check that the NotImplementedError raised by MyFoo.ping()
	// is returned by MyFoo.pong().
	ok := false
	a := NewDirectorFoo(&MyFoo{})
	b := Launder(a)
	func() {
		defer func() {
			e := recover()
			if e.(string) == "MyFoo::ping() EXCEPTION" {
				ok = true
			} else {
				panic("Unexpected error message: " + e.(string))
			}
		}()
		b.Pong()
	}()
	if !ok {
		panic(0)
	}

	// Check that if the method has the wrong return type it is
	// not called.
	ok = false
	a = NewDirectorFoo(&MyFoo2{})
	b = Launder(a)
	e := b.Pong()
	if e != "Foo::pong();"+"Foo::ping()" {
		panic(e)
	}

	// Check that the director can return an exception which
	// requires two arguments to the constructor, without mangling
	// it.
	ok = false
	a = NewDirectorFoo(&MyFoo3{})
	b = Launder(a)
	func() {
		defer func() {
			e := recover()
			if e.(*Exception).msg == "foobar" {
				ok = true
			} else {
				panic("Unexpected error message: " + e.(string))
			}
		}()
		b.Pong()
	}()
	if !ok {
		panic(0)
	}

	func() {
		defer func() {
			e := recover()
			_ = e.(Exception2)
		}()
		panic(NewException2())
	}()

	func() {
		defer func() {
			e := recover()
			_ = e.(Exception1)
		}()
		panic(NewException1())
	}()
}
