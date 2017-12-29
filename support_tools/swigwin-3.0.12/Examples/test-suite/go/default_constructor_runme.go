package main

import dc "./default_constructor"

func main() {
	a := dc.NewA()
	dc.DeleteA(a)

	aa := dc.NewAA()
	dc.DeleteAA(aa)

	cc := dc.NewCC()
	dc.DeleteCC(cc)

	e := dc.NewE()
	dc.DeleteE(e)

	ee := dc.NewEE()
	dc.DeleteEE(ee)

	f := dc.NewF()
	f.Destroy()

	g := dc.NewG()

	dc.GDestroy(g)

	gg := dc.NewGG()
	dc.DeleteGG(gg)

	dc.NewHH(1, 1)
}
