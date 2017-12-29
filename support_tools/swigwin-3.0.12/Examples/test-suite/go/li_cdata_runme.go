package main

import . "./li_cdata"

func main() {
	s := "ABC abc"
	m := Malloc(256)
	Memmove(m, s, len(s))
	ss := Cdata(m, 7)
	if string(ss) != "ABC abc" {
		panic("failed")
	}
}
