package main

import "mod_a"
import "mod_b"

func main() {
	c := mod_b.NewC()
	d := mod_b.NewD()
	d.DoSomething(mod_a.SwigcptrA(c.Swigcptr()))
}
