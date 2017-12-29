package main

import . "./director_string"

type B struct { // From A
	abi  A
	smem string
}

func NewB(s string) A {
	p := &B{nil, ""}
	ret := NewDirectorA(p, s)
	p.abi = ret
	return ret
}

func (p *B) Get_first() string {
	return DirectorAGet_first(p.abi) + " world!"
}

func (p *B) Process_text(s string) {
	DirectorAProcess_text(p.abi, s)
	p.smem = "hello"
}

func main() {
	b := NewB("hello")

	b.Get(0)
	if b.Get_first() != "hello world!" {
		panic(b.Get_first())
	}

	b.Call_process_func()

	if b.DirectorInterface().(*B).smem != "hello" {
		panic(b.DirectorInterface().(*B).smem)
	}
}
