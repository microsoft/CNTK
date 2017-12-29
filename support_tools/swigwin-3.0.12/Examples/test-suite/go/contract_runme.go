package main

import "./contract"

func main() {
	contract.Test_preassert(1, 2)
	contract.Test_postassert(3)
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Postassertions are broken")
			}
		}()
		contract.Test_postassert(-3)
	}()

	contract.Test_prepost(2, 3)
	contract.Test_prepost(5, -4)
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Preassertions are broken")
			}
		}()
		contract.Test_prepost(-3, 4)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Postassertions are broken")
			}
		}()
		contract.Test_prepost(4, -10)
	}()

	f := contract.NewFoo()
	f.Test_preassert(4, 5)
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Method preassertion.")
			}
		}()
		f.Test_preassert(-2, 3)
	}()

	f.Test_postassert(4)
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Method postassertion")
			}
		}()
		f.Test_postassert(-4)
	}()

	f.Test_prepost(3, 4)
	f.Test_prepost(4, -3)
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Method preassertion.")
			}
		}()
		f.Test_prepost(-4, 2)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Method postassertion.")
			}
		}()
		f.Test_prepost(4, -10)
	}()

	contract.FooStest_prepost(4, 0)
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Static method preassertion")
			}
		}()
		contract.FooStest_prepost(-4, 2)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Static method posteassertion")
			}
		}()
		contract.FooStest_prepost(4, -10)
	}()

	b := contract.NewBar()
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion.")
			}
		}()
		b.Test_prepost(2, -4)
	}()

	d := contract.NewD()
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Foo(-1, 1, 1, 1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Foo(1, -1, 1, 1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Foo(1, 1, -1, 1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Foo(1, 1, 1, -1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Foo(1, 1, 1, 1, -1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Bar(-1, 1, 1, 1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Bar(1, -1, 1, 1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Bar(1, 1, -1, 1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Bar(1, 1, 1, -1, 1)
	}()

	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! Inherited preassertion (D).")
			}
		}()
		d.Bar(1, 1, 1, 1, -1)
	}()

	//Namespace
	contract.NewMyClass(1)
	func() {
		defer func() {
			if recover() == nil {
				panic("Failed! constructor preassertion")
			}
		}()
		contract.NewMyClass(0)
	}()
}
