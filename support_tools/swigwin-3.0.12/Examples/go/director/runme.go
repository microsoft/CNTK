package main

import (
	"./example"
	"fmt"
	"os"
)

func Compare(name string, got string, exp string) error {
	fmt.Printf("%s; Got: '%s'; Expected: '%s'\n", name, got, exp)
	if got != exp {
		return fmt.Errorf("%s returned unexpected string! Got: '%s'; Expected: '%s'\n", name, got, exp)
	}
	return nil
}

func TestFooBarCpp() error {
	fb := example.NewFooBarCpp()
	defer example.DeleteFooBarCpp(fb)
	return Compare("FooBarCpp.FooBar()", fb.FooBar(), "C++ Foo, C++ Bar")
}

func TestFooBarGo() error {
	fb := example.NewFooBarGo()
	defer example.DeleteFooBarGo(fb)
	return Compare("FooBarGo.FooBar()", fb.FooBar(), "Go Foo, Go Bar")
}

func main() {
	fmt.Println("Test output:")
	fmt.Println("------------")
	err := TestFooBarCpp()
	err = TestFooBarGo()
	fmt.Println("------------")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Tests failed! Last error: %s\n", err.Error())
		os.Exit(1)
	}
}
