package main

import "strings"
import . "./exception_order"

func main() {
	a := NewA()

	func() {
		defer func() {
			e := recover()
			if strings.Index(e.(string), "E1") == -1 {
				panic(e.(string))
			}
		}()
		a.Foo()
	}()

	func() {
		defer func() {
			e := recover()
			if strings.Index(e.(string), "E2") == -1 {
				panic(e.(string))
			}
		}()
		a.Bar()
	}()

	func() {
		defer func() {
			e := recover()
			if e.(string) != "postcatch unknown" {
				panic("bad exception order")
			}
		}()
		a.Foobar()
	}()

	func() {
		defer func() {
			e := recover()
			if strings.Index(e.(string), "E1") == -1 {
				panic(e.(string))
			}
		}()
		a.Barfoo(1)
	}()

	func() {
		defer func() {
			e := recover()
			if strings.Index(e.(string), "E2") == -1 {
				panic(e.(string))
			}
		}()
		a.Barfoo(2)
	}()
}
