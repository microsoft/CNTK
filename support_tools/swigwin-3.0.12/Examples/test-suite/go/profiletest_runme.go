package main

import "fmt"
import "./profiletest"

func main() {
	a := profiletest.NewA()
	if false {
		fmt.Println(a)
	}

	b := profiletest.NewB()
	fn := func(a profiletest.A) profiletest.A { return b.Fn(a) }
	i := 50000
	for i != 0 {
		a = fn(a) //1
		a = fn(a) //2
		a = fn(a) //3
		a = fn(a) //4
		a = fn(a) //5
		a = fn(a) //6
		a = fn(a) //7
		a = fn(a) //8
		a = fn(a) //9
		a = fn(a) //10
		a = fn(a) //1
		a = fn(a) //2
		a = fn(a) //3
		a = fn(a) //4
		a = fn(a) //5
		a = fn(a) //6
		a = fn(a) //7
		a = fn(a) //8
		a = fn(a) //9
		a = fn(a) //20
		i -= 1
	}
}
