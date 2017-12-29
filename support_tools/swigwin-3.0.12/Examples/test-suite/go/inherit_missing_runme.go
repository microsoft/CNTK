package main

import "./inherit_missing"

func main() {
	a := inherit_missing.New_Foo()
	b := inherit_missing.NewBar()
	c := inherit_missing.NewSpam()

	x := inherit_missing.Do_blah(a)
	if x != "Foo::blah" {
		panic(x)
	}

	x = inherit_missing.Do_blah(b)
	if x != "Bar::blah" {
		panic(x)
	}

	x = inherit_missing.Do_blah(c)
	if x != "Spam::blah" {
		panic(x)
	}

	inherit_missing.Delete_Foo(a)
}
