// Test case from bug #1506850 "When threading is enabled, the
// interpreter will infinitely wait on a mutex the second time this
// type of extended method is called.  Attached is an example program
// that waits on the mutex to be unlocked."

package main

import . "./director_extend"

func main() {
	m := NewSpObject()
	if m.Dummy() != 666 {
		panic("1st call")
	}
	if m.Dummy() != 666 {
		panic("2nd call")
	}
}
