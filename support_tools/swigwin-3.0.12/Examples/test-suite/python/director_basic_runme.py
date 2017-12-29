import director_basic


class PyFoo(director_basic.Foo):

    def ping(self):
        return "PyFoo::ping()"


a = PyFoo()

if a.ping() != "PyFoo::ping()":
    raise RuntimeError, a.ping()

if a.pong() != "Foo::pong();PyFoo::ping()":
    raise RuntimeError, a.pong()

b = director_basic.Foo()

if b.ping() != "Foo::ping()":
    raise RuntimeError, b.ping()

if b.pong() != "Foo::pong();Foo::ping()":
    raise RuntimeError, b.pong()

a = director_basic.A1(1)

if a.rg(2) != 2:
    raise RuntimeError


class PyClass(director_basic.MyClass):

    def method(self, vptr):
        self.cmethod = 7
        pass

    def vmethod(self, b):
        b.x = b.x + 31
        return b


b = director_basic.Bar(3)
d = director_basic.MyClass()
c = PyClass()

cc = director_basic.MyClass_get_self(c)
dd = director_basic.MyClass_get_self(d)

bc = cc.cmethod(b)
bd = dd.cmethod(b)

cc.method(b)
if c.cmethod != 7:
    raise RuntimeError

if bc.x != 34:
    raise RuntimeError


if bd.x != 16:
    raise RuntimeError


class PyMulti(director_basic.Foo, director_basic.MyClass):

    def __init__(self):
        director_basic.Foo.__init__(self)
        director_basic.MyClass.__init__(self)
        pass

    def vmethod(self, b):
        b.x = b.x + 31
        return b

    def ping(self):
        return "PyFoo::ping()"

a = 0
for i in range(0, 100):
    pymult = PyMulti()
    pymult.pong()
    del pymult


pymult = PyMulti()


p1 = director_basic.Foo_get_self(pymult)
p2 = director_basic.MyClass_get_self(pymult)

p1.ping()
p2.vmethod(bc)
