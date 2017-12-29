package main

import "./li_attribute"

func main() {
	aa := li_attribute.NewA(1, 2, 3)

	if aa.GetA() != 1 {
		panic(0)
	}
	aa.SetA(3)
	if aa.GetA() != 3 {
		panic(aa.GetA())
	}

	if aa.GetB() != 2 {
		panic(aa.GetB())
	}
	aa.SetB(5)
	if aa.GetB() != 5 {
		panic(0)
	}

	if aa.GetD() != aa.GetB() {
		panic(0)
	}

	if aa.GetC() != 3 {
		panic(0)
	}

	pi := li_attribute.NewParam_i(7)
	if pi.GetValue() != 7 {
		panic(0)
	}
	pi.SetValue(3)
	if pi.GetValue() != 3 {
		panic(0)
	}

	b := li_attribute.NewB(aa)

	if b.GetA().GetC() != 3 {
		panic(0)
	}

	// class/struct attribute with get/set methods using
	// return/pass by reference
	myFoo := li_attribute.NewMyFoo()
	myFoo.SetX(8)
	myClass := li_attribute.NewMyClass()
	myClass.SetFoo(myFoo)
	if myClass.GetFoo().GetX() != 8 {
		panic(0)
	}

	// class/struct attribute with get/set methods using
	// return/pass by value
	myClassVal := li_attribute.NewMyClassVal()
	if myClassVal.GetReadWriteFoo().GetX() != -1 {
		panic(0)
	}
	if myClassVal.GetReadOnlyFoo().GetX() != -1 {
		panic(0)
	}
	myClassVal.SetReadWriteFoo(myFoo)
	if myClassVal.GetReadWriteFoo().GetX() != 8 {
		panic(0)
	}
	if myClassVal.GetReadOnlyFoo().GetX() != 8 {
		panic(0)
	}

	// string attribute with get/set methods using return/pass by
	// value
	myStringyClass := li_attribute.NewMyStringyClass("initial string")
	if myStringyClass.GetReadWriteString() != "initial string" {
		panic(0)
	}
	if myStringyClass.GetReadOnlyString() != "initial string" {
		panic(0)
	}
	myStringyClass.SetReadWriteString("changed string")
	if myStringyClass.GetReadWriteString() != "changed string" {
		panic(0)
	}
	if myStringyClass.GetReadOnlyString() != "changed string" {
		panic(0)
	}
}
