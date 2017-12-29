package main

import . "./template_ns4"

func main() {
	d := Make_Class_DD()
	if d.Test() != "test" {
		panic(0)
	}
}
