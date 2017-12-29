package main

import "./wrapmacro"

func main() {
	a := 2
	b := -1
	wrapmacro.Maximum(int64(a), int64(b))
	wrapmacro.Maximum(float64(a/7.0), float64(-b*256))
	wrapmacro.GUINT16_SWAP_LE_BE_CONSTANT(1)
}
