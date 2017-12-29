package main

import "./overload_polymorphic"

func main(){
	t := overload_polymorphic.NewDerived()
	
	if overload_polymorphic.Test(t) != 0 {
		panic("failed 1")
	}

	if overload_polymorphic.Test2(t) != 1 {
		panic("failed 2")
	}
}
