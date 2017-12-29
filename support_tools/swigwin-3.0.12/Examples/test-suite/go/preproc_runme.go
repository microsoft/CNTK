package main

import "./preproc"

func main() {
	if preproc.GetEndif() != 1 {
		panic(0)
	}

	if preproc.GetDefine() != 1 {
		panic(0)
	}

	if preproc.GetDefined() != 1 {
		panic(0)
	}

	if 2*preproc.One != preproc.Two {
		panic(0)
	}
}
