package main

import . "./fvirtual"

func main() {
	sw := NewNodeSwitch()
	n := NewNode()
	i := sw.AddChild(n)

	if i != 2 {
		panic("addChild")
	}
}
