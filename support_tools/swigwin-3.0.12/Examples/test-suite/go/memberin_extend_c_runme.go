package main

import "./memberin_extend_c"

func main() {
	t := memberin_extend_c.NewPerson()
	t.SetName("Fred Bloggs")
	if t.GetName() != "FRED BLOGGS" {
		panic("name wrong")
	}
}
