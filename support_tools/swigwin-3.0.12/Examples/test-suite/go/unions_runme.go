// This is the union runtime testcase. It ensures that values within a
// union embedded within a struct can be set and read correctly.

package main

import "./unions"

func main() {
	// Create new instances of SmallStruct and BigStruct for later use
	small := unions.NewSmallStruct()
	small.SetJill(200)

	big := unions.NewBigStruct()
	big.SetSmallstruct(small)
	big.SetJack(300)

	// Use SmallStruct then BigStruct to setup EmbeddedUnionTest.
	// Ensure values in EmbeddedUnionTest are set correctly for each.
	eut := unions.NewEmbeddedUnionTest()

	// First check the SmallStruct in EmbeddedUnionTest
	eut.SetNumber(1)
	eut.GetUni().SetSmall(small)
	Jill1 := eut.GetUni().GetSmall().GetJill()
	if Jill1 != 200 {
		panic("Runtime test1 failed")
	}

	Num1 := eut.GetNumber()
	if Num1 != 1 {
		panic("Runtime test2 failed")
	}

	// Secondly check the BigStruct in EmbeddedUnionTest
	eut.SetNumber(2)
	eut.GetUni().SetBig(big)
	Jack1 := eut.GetUni().GetBig().GetJack()
	if Jack1 != 300 {
		panic("Runtime test3 failed")
	}

	Jill2 := eut.GetUni().GetBig().GetSmallstruct().GetJill()
	if Jill2 != 200 {
		panic("Runtime test4 failed")
	}

	Num2 := eut.GetNumber()
	if Num2 != 2 {
		panic("Runtime test5 failed")
	}
}
