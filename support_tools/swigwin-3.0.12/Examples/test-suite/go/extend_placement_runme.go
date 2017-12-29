package main

import "./extend_placement"

func main() {
	foo := extend_placement.NewFoo()
	foo = extend_placement.NewFoo(1)
	foo = extend_placement.NewFoo(1, 1)
	foo.Spam()
	foo.Spam("hello")
	foo.Spam(1)
	foo.Spam(1, 1)
	foo.Spam(1, 1, 1)
	foo.Spam(extend_placement.NewFoo())
	foo.Spam(extend_placement.NewFoo(), 1.0)

	bar := extend_placement.NewBar()
	bar = extend_placement.NewBar(1)
	bar.Spam()
	bar.Spam("hello")
	bar.Spam(1)
	bar.Spam(1, 1)
	bar.Spam(1, 1, 1)
	bar.Spam(extend_placement.NewBar())
	bar.Spam(extend_placement.NewBar(), 1.0)

	footi := extend_placement.NewFooTi()
	footi = extend_placement.NewFooTi(1)
	footi = extend_placement.NewFooTi(1, 1)
	footi.Spam()
	footi.Spam("hello")
	footi.Spam(1)
	footi.Spam(1, 1)
	footi.Spam(1, 1, 1)
	footi.Spam(extend_placement.NewFoo())
	footi.Spam(extend_placement.NewFoo(), 1.0)

	barti := extend_placement.NewBarTi()
	barti = extend_placement.NewBarTi(1)
	barti.Spam()
	barti.Spam("hello")
	barti.Spam(1)
	barti.Spam(1, 1)
	barti.Spam(1, 1, 1)
	barti.Spam(extend_placement.NewBar())
	barti.Spam(extend_placement.NewBar(), 1.0)
}
