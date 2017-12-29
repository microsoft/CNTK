package main

import "./inctest"

func main() {
	inctest.NewA()
	inctest.NewB()

	// Check the import in subdirectory worked
	if inctest.Importtest1(5) != 15 {
		panic("import test 1 failed")
	}

	a := []byte("black")
	if inctest.Importtest2(string(a)) != "white" {
		panic("import test 2 failed")
	}
}
