package main

import "multi_import_a"
import "multi_import_b"

func main() {
	x := multi_import_b.NewXXX()
	if x.Testx() != 0 {
		panic(0)
	}

	y := multi_import_b.NewYYY()
	if y.Testx() != 0 {
		panic(0)
	}
	if y.Testy() != 1 {
		panic(0)
	}

	z := multi_import_a.NewZZZ()
	if z.Testx() != 0 {
		println("z.Testx", z.Testx(), z.Testz())
		panic(0)
	}
	if z.Testz() != 2 {
		println("z.Testz", z.Testz())
		panic(0)
	}
}
