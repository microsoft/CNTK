package main

import . "./using_composition"

func main() {
	f := NewFooBar()
	if f.Blah(3).(int) != 3 {
		panic("FooBar::blah(int)")
	}

	if f.Blah(3.5) != 3.5 {
		panic("FooBar::blah(double)")
	}

	if f.Blah("hello").(string) != "hello" {
		panic("FooBar::blah(char *)")
	}

	f2 := NewFooBar2()
	if f2.Blah(3).(int) != 3 {
		panic("FooBar2::blah(int)")
	}

	if f2.Blah(3.5) != 3.5 {
		panic("FooBar2::blah(double)")
	}

	if f2.Blah("hello").(string) != "hello" {
		panic("FooBar2::blah(char *)")
	}

	f3 := NewFooBar3()
	if f3.Blah(3).(int) != 3 {
		panic("FooBar3::blah(int)")
	}

	if f3.Blah(3.5) != 3.5 {
		panic("FooBar3::blah(double)")
	}

	if f3.Blah("hello").(string) != "hello" {
		panic("FooBar3::blah(char *)")
	}
}
