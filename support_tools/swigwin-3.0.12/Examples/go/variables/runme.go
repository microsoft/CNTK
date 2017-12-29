// This example illustrates global variable access from Go.

package main

import (
	"./example"
	"fmt"
)

func main() {
	// Try to set the values of some global variables

	example.SetIvar(42)
	example.SetSvar(-31000)
	example.SetLvar(65537)
	example.SetUivar(123456)
	example.SetUsvar(61000)
	example.SetUlvar(654321)
	example.SetScvar(-13)
	example.SetUcvar(251)
	example.SetCvar('S')
	example.SetFvar(3.14159)
	example.SetDvar(2.1828)
	example.SetStrvar("Hello World")
	example.SetIptrvar(example.New_int(37))
	example.SetPtptr(example.New_Point(37, 42))
	example.SetName("Bill")

	// Now print out the values of the variables

	fmt.Println("Variables (values printed from Go)")

	fmt.Println("ivar      =", example.GetIvar())
	fmt.Println("svar      =", example.GetSvar())
	fmt.Println("lvar      =", example.GetLvar())
	fmt.Println("uivar     =", example.GetUivar())
	fmt.Println("usvar     =", example.GetUsvar())
	fmt.Println("ulvar     =", example.GetUlvar())
	fmt.Println("scvar     =", example.GetScvar())
	fmt.Println("ucvar     =", example.GetUcvar())
	fmt.Println("fvar      =", example.GetFvar())
	fmt.Println("dvar      =", example.GetDvar())
	fmt.Printf("cvar      = %c\n", example.GetCvar())
	fmt.Println("strvar    =", example.GetStrvar())
	fmt.Println("cstrvar   =", example.GetCstrvar())
	fmt.Println("iptrvar   =", example.GetIptrvar())
	fmt.Println("name      =", example.GetName())
	fmt.Println("ptptr     =", example.GetPtptr(), example.Point_print(example.GetPtptr()))
	fmt.Println("pt        =", example.GetPt(), example.Point_print(example.GetPt()))

	fmt.Println("\nVariables (values printed from C)")

	example.Print_vars()

	// This line would not compile: since status is marked with
	// %immutable, there is no SetStatus function.
	// fmt.Println("\nNow I'm going to try and modify some read only variables")
	// example.SetStatus(0)

	fmt.Println("\nI'm going to try and update a structure variable.\n")

	example.SetPt(example.GetPtptr())

	fmt.Println("The new value is")
	example.Pt_print()
	fmt.Println("You should see the value", example.Point_print(example.GetPtptr()))
}
