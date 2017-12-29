package main

import . "./li_std_vector_ptr"
import "fmt"

func check(val1 int, val2 int) {
	if val1 != val2 {
		panic(fmt.Sprintf("Values are not the same %d %d", val1, val2))
	}
}
func main() {
	ip1 := MakeIntPtr(11)
	ip2 := MakeIntPtr(22)
	vi := NewIntPtrVector()
	vi.Add(ip1)
	vi.Add(ip2)
	check(GetValueFromVector(vi, 0), 11)
	check(GetValueFromVector(vi, 1), 22)
}
