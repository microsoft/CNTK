package main

import . "./li_cmalloc"

func main() {
	p := Malloc_int()
	Free_int(p)

	ok := false
	func() {
		defer func() {
			if recover() != nil {
				ok = true
			}
		}()
		p = Calloc_int(-1)
		if p == nil {
			ok = true
		}
		Free_int(p)
	}()
	if !ok {
		panic(0)
	}
}
