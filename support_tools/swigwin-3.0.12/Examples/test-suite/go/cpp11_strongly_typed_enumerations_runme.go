package main

import "fmt"
import . "./cpp11_strongly_typed_enumerations"

func enumCheck(actual int, expected int) int {
	if actual != expected {
		panic(fmt.Sprintf("Enum value mismatch. Expected: %d Actual: %d", expected, actual))
	}
	return expected + 1
}

func main() {
	var val = 0
	val = enumCheck(int(Enum1_Val1), val)
	val = enumCheck(int(Enum1_Val2), val)
	val = enumCheck(int(Enum1_Val3), 13)
	val = enumCheck(int(Enum1_Val4), val)
	val = enumCheck(int(Enum1_Val5a), 13)
	val = enumCheck(int(Enum1_Val6a), val)

	val = 0
	val = enumCheck(int(Enum2_Val1), val)
	val = enumCheck(int(Enum2_Val2), val)
	val = enumCheck(int(Enum2_Val3), 23)
	val = enumCheck(int(Enum2_Val4), val)
	val = enumCheck(int(Enum2_Val5b), 23)
	val = enumCheck(int(Enum2_Val6b), val)

	val = 0
	val = enumCheck(int(Val1), val)
	val = enumCheck(int(Val2), val)
	val = enumCheck(int(Val3), 43)
	val = enumCheck(int(Val4), val)

	val = 0
	val = enumCheck(int(Enum5_Val1), val)
	val = enumCheck(int(Enum5_Val2), val)
	val = enumCheck(int(Enum5_Val3), 53)
	val = enumCheck(int(Enum5_Val4), val)

	val = 0
	val = enumCheck(int(Enum6_Val1), val)
	val = enumCheck(int(Enum6_Val2), val)
	val = enumCheck(int(Enum6_Val3), 63)
	val = enumCheck(int(Enum6_Val4), val)

	val = 0
	val = enumCheck(int(Enum7td_Val1), val)
	val = enumCheck(int(Enum7td_Val2), val)
	val = enumCheck(int(Enum7td_Val3), 73)
	val = enumCheck(int(Enum7td_Val4), val)

	val = 0
	val = enumCheck(int(Enum8_Val1), val)
	val = enumCheck(int(Enum8_Val2), val)
	val = enumCheck(int(Enum8_Val3), 83)
	val = enumCheck(int(Enum8_Val4), val)

	val = 0
	val = enumCheck(int(Enum10_Val1), val)
	val = enumCheck(int(Enum10_Val2), val)
	val = enumCheck(int(Enum10_Val3), 103)
	val = enumCheck(int(Enum10_Val4), val)

	val = 0
	val = enumCheck(int(Class1Enum12_Val1), 1121)
	val = enumCheck(int(Class1Enum12_Val2), 1122)
	val = enumCheck(int(Class1Enum12_Val3), val)
	val = enumCheck(int(Class1Enum12_Val4), val)
	val = enumCheck(int(Class1Enum12_Val5c), 1121)
	val = enumCheck(int(Class1Enum12_Val6c), val)

	val = 0
	val = enumCheck(int(Class1Val1), 1131)
	val = enumCheck(int(Class1Val2), 1132)
	val = enumCheck(int(Class1Val3), val)
	val = enumCheck(int(Class1Val4), val)
	val = enumCheck(int(Class1Val5d), 1131)
	val = enumCheck(int(Class1Val6d), val)

	val = 0
	val = enumCheck(int(Class1Enum14_Val1), 1141)
	val = enumCheck(int(Class1Enum14_Val2), 1142)
	val = enumCheck(int(Class1Enum14_Val3), val)
	val = enumCheck(int(Class1Enum14_Val4), val)
	val = enumCheck(int(Class1Enum14_Val5e), 1141)
	val = enumCheck(int(Class1Enum14_Val6e), val)

	// Requires nested class support to work
	//val = 0
	//val = enumCheck(int(Class1Struct1Enum12_Val1), 3121)
	//val = enumCheck(int(Class1Struct1Enum12_Val2), 3122)
	//val = enumCheck(int(Class1Struct1Enum12_Val3), val)
	//val = enumCheck(int(Class1Struct1Enum12_Val4), val)
	//val = enumCheck(int(Class1Struct1Enum12_Val5f), 3121)
	//val = enumCheck(int(Class1Struct1Enum12_Val6f), val)
	//
	//val = 0
	//val = enumCheck(int(Class1Struct1Val1), 3131)
	//val = enumCheck(int(Class1Struct1Val2), 3132)
	//val = enumCheck(int(Class1Struct1Val3), val)
	//val = enumCheck(int(Class1Struct1Val4), val)
	//
	//val = 0
	//val = enumCheck(int(Class1Struct1Enum14_Val1), 3141)
	//val = enumCheck(int(Class1Struct1Enum14_Val2), 3142)
	//val = enumCheck(int(Class1Struct1Enum14_Val3), val)
	//val = enumCheck(int(Class1Struct1Enum14_Val4), val)
	//val = enumCheck(int(Class1Struct1Enum14_Val5g), 3141)
	//val = enumCheck(int(Class1Struct1Enum14_Val6g), val)

	val = 0
	val = enumCheck(int(Class2Enum12_Val1), 2121)
	val = enumCheck(int(Class2Enum12_Val2), 2122)
	val = enumCheck(int(Class2Enum12_Val3), val)
	val = enumCheck(int(Class2Enum12_Val4), val)
	val = enumCheck(int(Class2Enum12_Val5h), 2121)
	val = enumCheck(int(Class2Enum12_Val6h), val)

	val = 0
	val = enumCheck(int(Class2Val1), 2131)
	val = enumCheck(int(Class2Val2), 2132)
	val = enumCheck(int(Class2Val3), val)
	val = enumCheck(int(Class2Val4), val)
	val = enumCheck(int(Class2Val5i), 2131)
	val = enumCheck(int(Class2Val6i), val)

	val = 0
	val = enumCheck(int(Class2Enum14_Val1), 2141)
	val = enumCheck(int(Class2Enum14_Val2), 2142)
	val = enumCheck(int(Class2Enum14_Val3), val)
	val = enumCheck(int(Class2Enum14_Val4), val)
	val = enumCheck(int(Class2Enum14_Val5j), 2141)
	val = enumCheck(int(Class2Enum14_Val6j), val)

	// Requires nested class support to work
	//val = 0
	//val = enumCheck(int(Class2Struct1Enum12_Val1), 4121)
	//val = enumCheck(int(Class2Struct1Enum12_Val2), 4122)
	//val = enumCheck(int(Class2Struct1Enum12_Val3), val)
	//val = enumCheck(int(Class2Struct1Enum12_Val4), val)
	//val = enumCheck(int(Class2Struct1Enum12_Val5k), 4121)
	//val = enumCheck(int(Class2Struct1Enum12_Val6k), val)
	//
	//val = 0
	//val = enumCheck(int(Class2Struct1Val1), 4131)
	//val = enumCheck(int(Class2Struct1Val2), 4132)
	//val = enumCheck(int(Class2Struct1Val3), val)
	//val = enumCheck(int(Class2Struct1Val4), val)
	//val = enumCheck(int(Class2Struct1Val5l), 4131)
	//val = enumCheck(int(Class2Struct1Val6l), val)
	//
	//val = 0
	//val = enumCheck(int(Class2Struct1Enum14_Val1), 4141)
	//val = enumCheck(int(Class2Struct1Enum14_Val2), 4142)
	//val = enumCheck(int(Class2Struct1Enum14_Val3), val)
	//val = enumCheck(int(Class2Struct1Enum14_Val4), val)
	//val = enumCheck(int(Class2Struct1Enum14_Val5m), 4141)
	//val = enumCheck(int(Class2Struct1Enum14_Val6m), val)

	class1 := NewClass1()
	enumCheck(int(class1.Class1Test1(Enum1_Val5a)), 13)
	enumCheck(int(class1.Class1Test2(Class1Enum12_Val5c)), 1121)
	//enumCheck(int(class1.Class1Test3(Class1Struct1Enum12_Val5f)), 3121)

	enumCheck(int(GlobalTest1(Enum1_Val5a)), 13)
	enumCheck(int(GlobalTest2(Class1Enum12_Val5c)), 1121)
	//enumCheck(int(GlobalTest3(Class1Struct1Enum12_Val5f)), 3121)

}
