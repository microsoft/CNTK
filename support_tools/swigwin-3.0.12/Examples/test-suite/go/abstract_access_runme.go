package main

import "./abstract_access"

func main() {
	d := abstract_access.NewD()
	if d.Do_x() != 1 {
		panic(d.Do_x())
	}
}
