package main

import . "./director_protected"

type FooBar struct{} // From Bar
func (p *FooBar) Ping() string {
	return "FooBar::ping();"
}

type FooBar2 struct{} // From Bar
func (p *FooBar2) Ping() string {
	return "FooBar2::ping();"
}
func (p *FooBar2) Pang() string {
	return "FooBar2::pang();"
}

type FooBar3 struct{} // From Bar
func (p *FooBar3) Cheer() string {
	return "FooBar3::cheer();"
}

func main() {
	b := NewBar()
	f := b.Create()
	fb := NewDirectorBar(&FooBar{})
	fb2 := NewDirectorBar(&FooBar2{})
	fb3 := NewDirectorBar(&FooBar3{})

	s := fb.Used()
	if s != "Foo::pang();Bar::pong();Foo::pong();FooBar::ping();" {
		panic(0)
	}

	s = fb2.Used()
	if s != "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();" {
		panic(0)
	}

	s = b.Pong()
	if s != "Bar::pong();Foo::pong();Bar::ping();" {
		panic(0)
	}

	s = f.Pong()
	if s != "Bar::pong();Foo::pong();Bar::ping();" {
		panic(0)
	}

	s = fb.Pong()
	if s != "Bar::pong();Foo::pong();FooBar::ping();" {
		panic(0)
	}

	s = fb3.DirectorInterface().(*FooBar3).Cheer()
	if s != "FooBar3::cheer();" {
		panic(s)
	}
	if fb2.Callping() != "FooBar2::ping();" {
		panic("bad fb2.callping")
	}
	if fb2.Callcheer() != "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();" {
		panic("bad fb2.callcheer")
	}

	if fb3.Callping() != "Bar::ping();" {
		panic("bad fb3.callping")
	}

	if fb3.Callcheer() != "FooBar3::cheer();" {
		panic("bad fb3.callcheer")
	}
}
