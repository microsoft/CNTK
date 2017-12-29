package main

import "./director_detect"

type MyBar struct {
	val int
} // From director_detect.Bar

func NewMyBar() director_detect.Bar {
	return director_detect.NewDirectorBar(&MyBar{2})
}

func (p *MyBar) Get_value() int {
	p.val++
	return p.val
}

func (p *MyBar) Get_class() director_detect.A {
	p.val++
	return director_detect.NewA()
}

func (p *MyBar) Just_do_it() {
	p.val++
}

func (p *MyBar) Clone() director_detect.Bar {
	return director_detect.NewDirectorBar(&MyBar{p.val})
}

func main() {
	b := NewMyBar()

	f := b.Baseclass()

	v := f.Get_value()
	_ = f.Get_class()
	f.Just_do_it()

	c := b.DirectorInterface().(*MyBar).Clone()
	vc := c.Get_value()

	if (v != 3) || (b.DirectorInterface().(*MyBar).val != 5) || (vc != 6) {
		panic("Bad virtual detection")
	}
}
