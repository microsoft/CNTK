package main

import . "./smart_pointer_templatevariables"

func main() {
	d := NewDiffImContainerPtr_D(Create(1234, 5678))

	if d.GetId() != 1234 {
		panic(0)
	}
	//if (d.xyz != 5678):
	//  panic(0)

	d.SetId(4321)
	//d.xyz = 8765

	if d.GetId() != 4321 {
		panic(0)
	}
	//if (d.xyz != 8765):
	//  panic(0)
}
