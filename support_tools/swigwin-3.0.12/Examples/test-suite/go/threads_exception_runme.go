package main

import "strings"
import "./threads_exception"

func main() {
	t := threads_exception.NewTest()

	error := true
	func() {
		defer func() {
			error = recover() == nil
		}()
		t.Unknown()
	}()
	if error {
		panic(0)
	}

	error = true
	func() {
		defer func() {
			error = strings.Index(recover().(string), "int exception") == -1
		}()
		t.Simple()
	}()
	if error {
		panic(0)
	}

	error = true
	func() {
		defer func() {
			error = recover().(string) != "I died."
		}()
		t.Message()
	}()
	if error {
		panic(0)
	}

	error = true
	func() {
		defer func() {
			e := recover().(string)
			error = strings.Index(e, "Exc exception") == -1
		}()
		t.Hosed()
	}()
	if error {
		panic(0)
	}

	for i := 1; i < 4; i++ {
		error = true
		func() {
			defer func() {
				error = recover() == nil
			}()
			t.Multi(i)
		}()
		if error {
			panic(0)
		}
	}
}
