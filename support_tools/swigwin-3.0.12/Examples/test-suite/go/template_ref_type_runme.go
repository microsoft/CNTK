package main

import "./template_ref_type"

func main() {
	xr := template_ref_type.NewXC()
	y := template_ref_type.NewY()
	y.Find(xr)
}
