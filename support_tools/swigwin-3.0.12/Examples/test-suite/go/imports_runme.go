// This is the import runtime testcase.

package main

import "imports_b"
import "imports_a"

func main() {
	x := imports_b.NewB()
	x.Hello()

	_ = imports_a.NewA()

	c := imports_b.NewC()
	_ = c.Get_a(c)
	_ = c.Get_a_type(c)
}
