package main

import . "./li_cpointer"

func main() {
	p := New_intp()
	Intp_assign(p, 3)

	if Intp_value(p) != 3 {
		panic(0)
	}

	Delete_intp(p)
}
