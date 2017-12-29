// Note: This example assumes that namespaces are flattened
package main

import "./cpp_namespace"

func main() {
	n := cpp_namespace.Fact(4)
	if n != 24 {
		panic("Bad return value!")
	}

	if cpp_namespace.GetFoo() != 42 {
		panic("Bad variable value!")
	}

	t := cpp_namespace.NewTest()
	if t.Method() != "Test::method" {
		panic("Bad method return value!")
	}

	if cpp_namespace.Do_method(t) != "Test::method" {
		panic("Bad return value!")
	}

	if cpp_namespace.Do_method2(t) != "Test::method" {
		panic("Bad return value!")
	}

	cpp_namespace.Weird("hello", 4)

	cpp_namespace.DeleteTest(t)

	t2 := cpp_namespace.NewTest2()
	t3 := cpp_namespace.NewTest3()
	t4 := cpp_namespace.NewTest4()
	t5 := cpp_namespace.NewTest5()

	if cpp_namespace.Foo3(42) != 42 {
		panic("Bad return value!")
	}

	if cpp_namespace.Do_method3(t2, 40) != "Test2::method" {
		panic("Bad return value!")
	}

	if cpp_namespace.Do_method3(t3, 40) != "Test3::method" {
		panic("Bad return value!")
	}

	if cpp_namespace.Do_method3(t4, 40) != "Test4::method" {
		panic("Bad return value!")
	}

	if cpp_namespace.Do_method3(t5, 40) != "Test5::method" {
		panic("Bad return value!")
	}
}
