package main

import (
	. "./example"
	"fmt"
)

func main() {
	fmt.Println("Adding and calling a normal C++ callback")
	fmt.Println("----------------------------------------")

	caller := NewCaller()
	callback := NewCallback()

	caller.SetCallback(callback)
	caller.Call()
	caller.DelCallback()

	go_callback := NewGoCallback()

	fmt.Println()
	fmt.Println("Adding and calling a Go callback")
	fmt.Println("--------------------------------")

	caller.SetCallback(go_callback)
	caller.Call()
	caller.DelCallback()

	DeleteGoCallback(go_callback)

	fmt.Println()
	fmt.Println("Go exit")
}
