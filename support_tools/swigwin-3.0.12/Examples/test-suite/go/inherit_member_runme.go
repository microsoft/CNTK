package main

import wrap "./inherit_member"

func main() {
	s := wrap.NewChild()
	s.SetPvar("p")
	s.SetCvar("c")
	if s.GetPvar() != "p" {
		panic(s.GetPvar())
	}
	if s.GetCvar() != "c" {
		panic(s.GetCvar())
	}
}
