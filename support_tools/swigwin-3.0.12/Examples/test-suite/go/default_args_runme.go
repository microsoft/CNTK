package main

import "./default_args"

func main() {
	if default_args.StaticsStaticmethod() != 60 {
		panic(0)
	}

	if default_args.Cfunc1(1) != 2 {
		panic(0)
	}

	if default_args.Cfunc2(1) != 3 {
		panic(0)
	}

	if default_args.Cfunc3(1) != 4 {
		panic(0)
	}

	f := default_args.NewFoo()

	f.Newname()
	f.Newname(1)
}
