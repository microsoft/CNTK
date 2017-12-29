package main

import "./cpp_enum"

func main() {
	f := cpp_enum.NewFoo()

	if f.GetHola() != cpp_enum.FooHello {
		panic(f.GetHola())
	}

	f.SetHola(cpp_enum.FooHi)
	if f.GetHola() != cpp_enum.FooHi {
		panic(f.GetHola())
	}

	f.SetHola(cpp_enum.FooHello)

	if f.GetHola() != cpp_enum.FooHello {
		panic(f.GetHola())
	}

	cpp_enum.SetHi(cpp_enum.Hello)
	if cpp_enum.GetHi() != cpp_enum.Hello {
		panic(cpp_enum.Hi)
	}
}
