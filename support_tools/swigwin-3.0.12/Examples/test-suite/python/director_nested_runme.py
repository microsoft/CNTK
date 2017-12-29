from director_nested import *


class A(FooBar_int):

    def do_step(self):
        return "A::do_step;"

    def get_value(self):
        return "A::get_value"

    pass


a = A()
if a.step() != "Bar::step;Foo::advance;Bar::do_advance;A::do_step;":
    raise RuntimeError, "Bad A virtual resolution"


class B(FooBar_int):

    def do_advance(self):
        return "B::do_advance;" + self.do_step()

    def do_step(self):
        return "B::do_step;"

    def get_value(self):
        return 1

    pass


b = B()

if b.step() != "Bar::step;Foo::advance;B::do_advance;B::do_step;":
    raise RuntimeError, "Bad B virtual resolution"


class C(FooBar_int):

    def do_advance(self):
        return "C::do_advance;" + FooBar_int.do_advance(self)

    def do_step(self):
        return "C::do_step;"

    def get_value(self):
        return 2

    def get_name(self):
        return FooBar_int.get_name(self) + " hello"

    pass

cc = C()
c = FooBar_int_get_self(cc)
c.advance()

if c.get_name() != "FooBar::get_name hello":
    raise RuntimeError

if c.name() != "FooBar::get_name hello":
    raise RuntimeError
