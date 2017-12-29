package main

import "./enum_template"

func main() {
	if enum_template.MakeETest() != 1 {
		panic(0)
	}

	enum_template.TakeETest(0)
}
