package main

import . "./namespace_typemap"

func main() {
	if Stest1("hello") != "hello" {
		panic(0)
	}

	if Stest2("hello") != "hello" {
		panic(0)
	}

	if Stest3("hello") != "hello" {
		panic(0)
	}

	if Stest4("hello") != "hello" {
		panic(0)
	}

	if Stest5("hello") != "hello" {
		panic(0)
	}

	if Stest6("hello") != "hello" {
		panic(0)
	}

	if Stest7("hello") != "hello" {
		panic(0)
	}

	if Stest8("hello") != "hello" {
		panic(0)
	}

	if Stest9("hello") != "hello" {
		panic(0)
	}

	if Stest10("hello") != "hello" {
		panic(0)
	}

	if Stest11("hello") != "hello" {
		panic(0)
	}

	if Stest12("hello") != "hello" {
		panic(0)
	}

	c := complex(2, 3)
	r := real(c)

	if Ctest1(c) != r {
		println(Ctest1(c))
		panic(Ctest1(c))
	}

	if Ctest2(c) != r {
		panic(0)
	}

	if Ctest3(c) != r {
		panic(0)
	}

	if Ctest4(c) != r {
		panic(0)
	}

	if Ctest5(c) != r {
		panic(0)
	}

	if Ctest6(c) != r {
		panic(0)
	}

	if Ctest7(c) != r {
		panic(0)
	}

	if Ctest8(c) != r {
		panic(0)
	}

	if Ctest9(c) != r {
		panic(0)
	}

	if Ctest10(c) != r {
		panic(0)
	}

	if Ctest11(c) != r {
		panic(0)
	}

	if Ctest12(c) != r {
		panic(0)
	}

	ok := false
	func() {
		defer func() {
			ok = recover() != nil
		}()
		Ttest1(-14)
	}()
	if !ok {
		panic(0)
	}
}
