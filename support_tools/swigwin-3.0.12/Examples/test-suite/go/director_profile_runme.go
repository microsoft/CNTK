package main

import "fmt"
import "./director_profile"

type MyB struct{} // From director_profile.B
func (p *MyB) Vfi(a int) int {
	return a + 3
}

func main() {
	_ = director_profile.NewA()
	myb := director_profile.NewDirectorB(&MyB{})
	b := director_profile.BGet_self(myb)

	fi := func(a int) int {
		return b.Fi(a)
	}

	i := 50000
	a := 1
	for i != 0 {
		a = fi(a) // 1
		a = fi(a) // 2
		a = fi(a) // 3
		a = fi(a) // 4
		a = fi(a) // 5
		a = fi(a) // 6
		a = fi(a) // 7
		a = fi(a) // 8
		a = fi(a) // 9
		a = fi(a) // 10
		a = fi(a) // 1
		a = fi(a) // 2
		a = fi(a) // 3
		a = fi(a) // 4
		a = fi(a) // 5
		a = fi(a) // 6
		a = fi(a) // 7
		a = fi(a) // 8
		a = fi(a) // 9
		a = fi(a) // 20
		i -= 1
	}

	if false {
		fmt.Println(a)
	}
}
