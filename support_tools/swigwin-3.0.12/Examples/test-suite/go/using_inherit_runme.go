package main

import . "./using_inherit"

func main() {
	b := NewBar()
	if b.Test(3).(int) != 3 {
		panic("Bar::test(int)")
	}

	if b.Test(3.5).(float64) != 3.5 {
		panic("Bar::test(double)")
	}

	b2 := NewBar2()
	if b2.Test(3).(int) != 6 {
		panic("Bar2::test(int)")
	}

	if b2.Test(3.5).(float64) != 7.0 {
		panic("Bar2::test(double)")
	}

	b3 := NewBar3()
	if b3.Test(3).(int) != 6 {
		panic("Bar3::test(int)")
	}

	if b3.Test(3.5).(float64) != 7.0 {
		panic("Bar3::test(double)")
	}

	b4 := NewBar4()
	if b4.Test(3).(int) != 6 {
		panic("Bar4::test(int)")
	}

	if b4.Test(3.5).(float64) != 7.0 {
		panic("Bar4::test(double)")
	}

	bf1 := NewFred1()
	if bf1.Test(3).(int) != 3 {
		panic("Fred1::test(int)")
	}

	if bf1.Test(3.5).(float64) != 7.0 {
		panic("Fred1::test(double)")
	}

	bf2 := NewFred2()
	if bf2.Test(3).(int) != 3 {
		panic("Fred2::test(int)")
	}

	if bf2.Test(3.5).(float64) != 7.0 {
		panic("Fred2::test(double)")
	}
}
