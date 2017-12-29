package main

import "fmt"
import . "./rename_simple"

func main() {
	s := NewNewStruct()
	check(111, s.GetNewInstanceVariable(), "NewInstanceVariable")
	check(222, s.NewInstanceMethod(), "NewInstanceMethod")
	check(333, NewStructNewStaticMethod(), "NewStaticMethod")
	check(444, GetNewStructNewStaticVariable(), "NewStaticVariable")
	check(555, NewFunction(), "NewFunction")
	check(666, GetNewGlobalVariable(), "NewGlobalVariable")

	s.SetNewInstanceVariable(1111)
	SetNewStructNewStaticVariable(4444)
	SetNewGlobalVariable(6666)

	check(1111, s.GetNewInstanceVariable(), "NewInstanceVariable")
	check(4444, GetNewStructNewStaticVariable(), "NewStaticVariable")
	check(6666, GetNewGlobalVariable(), "NewGlobalVariable")
}

func check(expected, actual int, msg string) {
	if expected != actual {
		panic("Failed: Expected: " + fmt.Sprint(expected) +
			" actual: " + fmt.Sprint(actual) + " " + msg)
	}
}
