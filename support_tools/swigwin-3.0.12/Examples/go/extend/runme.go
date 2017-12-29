// This file illustrates the cross language polymorphism using directors.

package main

import (
	. "./example"
	"fmt"
)

func main() {
	// Create an instance of CEO, a class derived from the Go
	// proxy of the underlying C++ class.  The calls to getName()
	// and getPosition() are standard, the call to getTitle() uses
	// the director wrappers to call CEO.getPosition().
	e := NewCEO("Alice")
	fmt.Println(e.GetName(), " is a ", e.GetPosition())
	fmt.Println("Just call her \"", e.GetTitle(), "\"")
	fmt.Println("----------------------")

	// Create a new EmployeeList instance.  This class does not
	// have a C++ director wrapper, but can be used freely with
	// other classes that do.
	list := NewEmployeeList()

	// EmployeeList owns its items, so we must surrender ownership
	// of objects we add.
	// e.DisownMemory()
	list.AddEmployee(e)
	fmt.Println("----------------------")

	// Now we access the first four items in list (three are C++
	// objects that EmployeeList's constructor adds, the last is
	// our CEO).  The virtual methods of all these instances are
	// treated the same.  For items 0, 1, and 2, all methods
	// resolve in C++.  For item 3, our CEO, GetTitle calls
	// GetPosition which resolves in Go.  The call to GetPosition
	// is slightly different, however, because of the overridden
	// GetPosition() call, since now the object reference has been
	// "laundered" by passing through EmployeeList as an
	// Employee*.  Previously, Go resolved the call immediately in
	// CEO, but now Go thinks the object is an instance of class
	// Employee.  So the call passes through the Employee proxy
	// class and on to the C wrappers and C++ director, eventually
	// ending up back at the Go CEO implementation of
	// getPosition().  The call to GetTitle() for item 3 runs the
	// C++ Employee::getTitle() method, which in turn calls
	// GetPosition().  This virtual method call passes down
	// through the C++ director class to the Go implementation
	// in CEO.  All this routing takes place transparently.
	fmt.Println("(position, title) for items 0-3:")
	fmt.Println("  ", list.Get_item(0).GetPosition(), ", \"", list.Get_item(0).GetTitle(), "\"")
	fmt.Println("  ", list.Get_item(1).GetPosition(), ", \"", list.Get_item(1).GetTitle(), "\"")
	fmt.Println("  ", list.Get_item(2).GetPosition(), ", \"", list.Get_item(2).GetTitle(), "\"")
	fmt.Println("  ", list.Get_item(3).GetPosition(), ", \"", list.Get_item(3).GetTitle(), "\"")
	fmt.Println("----------------------")

	// Time to delete the EmployeeList, which will delete all the
	// Employee* items it contains. The last item is our CEO,
	// which gets destroyed as well and hence there is no need to
	// call DeleteCEO.
	DeleteEmployeeList(list)
	fmt.Println("----------------------")

	// All done.
	fmt.Println("Go exit")
}
