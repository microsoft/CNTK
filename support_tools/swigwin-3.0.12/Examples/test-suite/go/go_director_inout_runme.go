package main

import (
	wrap "./go_director_inout"
)

type GoMyClass struct {}

func (p *GoMyClass) Adjust(m map[string]interface{}) wrap.GoRetStruct {
	s := ""
	for k, v := range m {
		s += k + "," + v.(string) + ";"
	}
	return wrap.GoRetStruct{s}
}

func main() {
	a := wrap.NewDirectorMyClass(&GoMyClass{})
	m := map[string]interface{}{
		"first": "second",
	}
	s := a.Adjust(m)
	if s.Str != "first,second;" {
		panic(s)
	}

	a = wrap.NewDirectorMyClass(nil)
	s = a.Adjust(m)
	if s.Str != `{"first":"second"}` {
		panic(s.Str)
	}
}
