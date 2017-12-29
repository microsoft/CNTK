package main

import "./overload_rename"

func main() {
	_ = overload_rename.NewFoo(float32(1))
	_ = overload_rename.NewFoo(float32(1), float32(1))
	_ = overload_rename.NewFoo_int(float32(1), 1)
	_ = overload_rename.NewFoo_int(float32(1), 1, float32(1))
}
