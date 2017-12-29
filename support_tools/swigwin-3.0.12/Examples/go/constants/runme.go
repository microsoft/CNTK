package main

import (
	"./example"
	"fmt"
)

func main() {
	fmt.Println("ICONST  = ", example.ICONST, " (should be 42)")
	fmt.Println("FCONST  = ", example.FCONST, " (should be 2.1828)")
	fmt.Printf("CCONST  = %c (should be 'x')\n", example.CCONST)
	fmt.Printf("CCONST2 = %c(this should be on a new line)\n", example.CCONST2)
	fmt.Println("SCONST  = ", example.SCONST, " (should be 'Hello World')")
	fmt.Println("SCONST2 = ", example.SCONST2, " (should be '\"Hello World\"')")
	fmt.Println("EXPR    = ", example.EXPR, " (should be 48.5484)")
	fmt.Println("iconst  = ", example.Iconst, " (should be 37)")
	fmt.Println("fconst  = ", example.Fconst, " (should be 3.14)")
}
