package main

import "./abstract_typedef"

func main() {
	e := abstract_typedef.NewEngine()
	a := abstract_typedef.NewA()
	if !a.Write(e) {
		panic("failed")
	}
}
