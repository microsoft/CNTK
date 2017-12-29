package main

import (
	"encoding/json"
	"fmt"
	"reflect"

	"./go_inout"
)

type S struct {
	A int
	B string
	C float64
}

func (p *S) MarshalJSON() ([]byte, error) {
	return json.Marshal(*p)
}

func main() {
	v := &S{12, "hi", 34.5}
	m := go_inout.Same(v)
	want := map[string]interface{}{
		// The type of A changes from int to float64 because
		// JSON has no ints.
		"A": float64(12),
		"B": "hi",
		"C": 34.5,
	}
	if !reflect.DeepEqual(m, want) {
		fmt.Println("got", m, "want", want)
		panic(m)
	}

	a := []string{"a", "bc", "def"}
	go_inout.DoubleArray(&a)
	dwant := []string{"a", "bc", "def", "aa", "bcbc", "defdef"}
	if !reflect.DeepEqual(a, dwant) {
		fmt.Println("got", a, "want", dwant)
		panic(a)
	}

	c2 := go_inout.NewC2()
	pm := c2.M()
	want = map[string]interface{}{
		"ID": float64(1),
	}
	if !reflect.DeepEqual(*pm, want) {
		fmt.Println("for c2.M got", pm, "want", want)
		panic(pm)
	}
}
