package main

import "fmt"
import "./minherit"

func main() {
	a := minherit.NewFoo()
	b := minherit.NewBar()
	c := minherit.NewFooBar()
	d := minherit.NewSpam()

	if a.Xget() != 1 {
		panic("1 Bad attribute value")
	}

	if b.Yget() != 2 {
		panic("2 Bad attribute value")
	}

	if c.Xget() != 1 || c.Yget() != 2 || c.Zget() != 3 {
		panic("3 Bad attribute value")
	}

	if d.Xget() != 1 || d.Yget() != 2 || d.Zget() != 3 || d.Wget() != 4 {
		panic("4 Bad attribute value")
	}

	if minherit.Xget(a) != 1 {
		panic(fmt.Sprintf("5 Bad attribute value %d", minherit.Xget(a)))
	}

	if minherit.Yget(b) != 2 {
		panic(fmt.Sprintf("6 Bad attribute value %d", minherit.Yget(b)))
	}

	if minherit.Xget(c) != 1 || minherit.Yget(c.SwigGetBar()) != 2 || minherit.Zget(c) != 3 {
		panic(fmt.Sprintf("7 Bad attribute value %d %d %d", minherit.Xget(c), minherit.Yget(c.SwigGetBar()), minherit.Zget(c)))
	}

	if minherit.Xget(d) != 1 || minherit.Yget(d.SwigGetBar()) != 2 || minherit.Zget(d) != 3 || minherit.Wget(d) != 4 {
		panic(fmt.Sprintf("8 Bad attribute value %d %d %d %d", minherit.Xget(d), minherit.Yget(d.SwigGetBar()), minherit.Zget(d), minherit.Wget(d)))
	}

	// Cleanse all of the pointers and see what happens

	aa := minherit.ToFooPtr(a)
	bb := minherit.ToBarPtr(b)
	cc := minherit.ToFooBarPtr(c)
	dd := minherit.ToSpamPtr(d)

	if aa.Xget() != 1 {
		panic("9 Bad attribute value")
	}

	if bb.Yget() != 2 {
		panic("10 Bad attribute value")
	}

	if cc.Xget() != 1 || cc.Yget() != 2 || cc.Zget() != 3 {
		panic("11 Bad attribute value")
	}

	if dd.Xget() != 1 || dd.Yget() != 2 || dd.Zget() != 3 || dd.Wget() != 4 {
		panic("12 Bad attribute value")
	}

	if minherit.Xget(aa) != 1 {
		panic(fmt.Sprintf("13 Bad attribute value %d", minherit.Xget(aa)))
	}

	if minherit.Yget(bb) != 2 {
		panic(fmt.Sprintf("14 Bad attribute value %d", minherit.Yget(bb)))
	}

	if minherit.Xget(cc) != 1 || minherit.Yget(cc.SwigGetBar()) != 2 || minherit.Zget(cc) != 3 {
		panic(fmt.Sprintf("15 Bad attribute value %d %d %d", minherit.Xget(cc), minherit.Yget(cc.SwigGetBar()), minherit.Zget(cc)))
	}

	if minherit.Xget(dd) != 1 || minherit.Yget(dd.SwigGetBar()) != 2 || minherit.Zget(dd) != 3 || minherit.Wget(dd) != 4 {
		panic(fmt.Sprintf("16 Bad attribute value %d %d %d %d", minherit.Xget(dd), minherit.Yget(dd.SwigGetBar()), minherit.Zget(dd), minherit.Wget(dd)))
	}
}
