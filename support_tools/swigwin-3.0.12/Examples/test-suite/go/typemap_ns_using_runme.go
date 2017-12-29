package main

import "./typemap_ns_using"

func main() {
	if typemap_ns_using.Spam(37) != 37 {
		panic(0)
	}
}
