package main

import . "./char_binary"

func main() {
	t := NewTest()
	if t.Strlen("hile") != 4 {
		print(t.Strlen("hile"))
		panic("bad multi-arg typemap")
	}

	if t.Strlen("hil\000") != 4 {
		panic("bad multi-arg typemap")
	}

	// creating a raw char*
	pc := New_pchar(5)
	Pchar_setitem(pc, 0, 'h')
	Pchar_setitem(pc, 1, 'o')
	Pchar_setitem(pc, 2, 'l')
	Pchar_setitem(pc, 3, 'a')
	Pchar_setitem(pc, 4, 0)

	Delete_pchar(pc)
}
