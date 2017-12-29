package main

import . "./template_tbase_template"

func main() {
	a := Make_Class_dd()
	if a.Test() != "test" {
		panic(0)
	}
}
