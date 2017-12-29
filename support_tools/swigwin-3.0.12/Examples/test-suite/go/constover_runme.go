package main

import (
	"./constover"
	"fmt"
	"os"
)

func main() {
	error := 0

	p := constover.Test("test")
	if p != "test" {
		fmt.Println("test failed!")
		error = 1
	}

	p = constover.Test_pconst("test")
	if p != "test_pconst" {
		fmt.Println("test_pconst failed!")
		error = 1
	}

	f := constover.NewFoo()
	p = f.Test("test")
	if p != "test" {
		fmt.Println("member-test failed!")
		error = 1
	}

	p = f.Test_pconst("test")
	if p != "test_pconst" {
		fmt.Println("member-test_pconst failed!")
		error = 1
	}

	p = f.Test_constm("test")
	if p != "test_constmethod" {
		fmt.Println("member-test_constm failed!")
		error = 1
	}

	p = f.Test_pconstm("test")
	if p != "test_pconstmethod" {
		fmt.Println("member-test_pconstm failed!")
		error = 1
	}

	os.Exit(error)
}
