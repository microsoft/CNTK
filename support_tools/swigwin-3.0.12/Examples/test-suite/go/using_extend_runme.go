package main

import . "./using_extend"

func main() {
	f := NewFooBar()
	if f.Blah(3).(int) != 3 {
		panic("blah(int)")
	}

	if f.Blah(3.5) != 3.5 {
		panic("blah(double)")
	}

	if f.Blah("hello").(string) != "hello" {
		panic("blah(char *)")
	}

	if f.Blah(3, 4).(int) != 7 {
		panic("blah(int,int)")
	}

	if f.Blah(3.5, 7.5) != (3.5 + 7.5) {
		panic("blah(double,double)")
	}

	if f.Duh(3) != 3 {
		panic("duh(int)")
	}
}
