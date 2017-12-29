package main

import "fmt"
import . "./director_classic"

type TargetLangPerson struct{} // From Person
func (p *TargetLangPerson) Id() string {
	return "TargetLangPerson"
}

type TargetLangChild struct{} // Form Child
func (p *TargetLangChild) Id() string {
	return "TargetLangChild"
}

type TargetLangGrandChild struct{} // From Grandchild
func (p *TargetLangGrandChild) Id() string {
	return "TargetLangGrandChild"
}

// Semis - don't override id() in target language

type TargetLangSemiPerson struct{} // From Person

type TargetLangSemiChild struct{} // From Child

type TargetLangSemiGrandChild struct{} // From GrandChild

// Orphans - don't override id() in C++

type TargetLangOrphanPerson struct{} // From OrphanPerson
func (p *TargetLangOrphanPerson) Id() string {
	return "TargetLangOrphanPerson"
}

type TargetLangOrphanChild struct{} // From OrphanChild
func (p *TargetLangOrphanChild) Id() string {
	return "TargetLangOrphanChild"
}

func check(person Person, expected string) {
	debug := false

	// Normal target language polymorphic call
	ret := person.Id()
	if debug {
		fmt.Println(ret)
	}
	if ret != expected {
		panic("Failed. Received: " + ret + " Expected: " + expected)
	}

	// Polymorphic call from C++
	caller := NewCaller()
	caller.SetCallback(person)
	ret = caller.Call()
	if debug {
		fmt.Println(ret)
	}
	if ret != expected {
		panic("Failed. Received: " + ret + " Expected: " + expected)
	}

	// Polymorphic call of object created in target language and
	// passed to C++ and back again
	baseclass := caller.BaseClass()
	ret = baseclass.Id()
	if debug {
		fmt.Println(ret)
	}
	if ret != expected {
		panic("Failed. Received: " + ret + " Expected: " + expected)
	}

	caller.ResetCallback()
	if debug {
		fmt.Println("----------------------------------------")
	}
}

func main() {
	person := NewPerson()
	check(person, "Person")
	DeletePerson(person)

	person = NewChild()
	check(person, "Child")
	DeletePerson(person)

	person = NewGrandChild()
	check(person, "GrandChild")
	DeletePerson(person)

	person = NewDirectorPerson(&TargetLangPerson{})
	check(person, "TargetLangPerson")
	DeleteDirectorPerson(person)

	person = NewDirectorChild(&TargetLangChild{})
	check(person, "TargetLangChild")
	DeleteDirectorChild(person.(Child))

	person = NewDirectorGrandChild(&TargetLangGrandChild{})
	check(person, "TargetLangGrandChild")
	DeleteDirectorGrandChild(person.(GrandChild))

	// Semis - don't override id() in target language
	person = NewDirectorPerson(&TargetLangSemiPerson{})
	check(person, "Person")
	DeleteDirectorPerson(person)

	person = NewDirectorChild(&TargetLangSemiChild{})
	check(person, "Child")
	DeleteDirectorChild(person.(Child))

	person = NewDirectorGrandChild(&TargetLangSemiGrandChild{})
	check(person, "GrandChild")
	DeleteDirectorGrandChild(person.(GrandChild))

	// Orphans - don't override id() in C++
	person = NewOrphanPerson()
	check(person, "Person")
	DeleteOrphanPerson(person.(OrphanPerson))

	person = NewOrphanChild()
	check(person, "Child")
	DeleteOrphanChild(person.(OrphanChild))

	person = NewDirectorOrphanPerson(&TargetLangOrphanPerson{})
	check(person, "TargetLangOrphanPerson")
	DeleteDirectorOrphanPerson(person.(OrphanPerson))

	person = NewDirectorOrphanChild(&TargetLangOrphanChild{})
	check(person, "TargetLangOrphanChild")
	DeleteDirectorOrphanChild(person.(OrphanChild))
}
