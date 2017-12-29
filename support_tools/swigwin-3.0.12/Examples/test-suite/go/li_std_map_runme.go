package main

import "./li_std_map"

func main() {
	a1 := li_std_map.NewA(3)
	a2 := li_std_map.NewA(7)

	_ = li_std_map.NewPairii(1, 2)
	p1 := li_std_map.NewPairA(1, a1)
	m := li_std_map.NewMapA()
	m.Set(1, a1)
	m.Set(2, a2)

	_ = li_std_map.P_identa(p1)
	_ = li_std_map.M_identa(m)

	m = li_std_map.NewMapA()
	m.Set(1, a1)
	m.Set(2, a2)

	mii := li_std_map.NewIntIntMap()

	mii.Set(1, 1)
	mii.Set(1, 2)

	if mii.Get(1) != 2 {
		panic(0)
	}
}
