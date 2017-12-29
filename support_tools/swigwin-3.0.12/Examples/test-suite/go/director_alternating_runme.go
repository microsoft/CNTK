package main

import . "./director_alternating"

func main() {
	id := GetBar().Id()
	if id != IdFromGetBar() {
		panic(id)
	}
}
