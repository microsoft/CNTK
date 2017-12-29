import director_property


class PyFoo(director_property.Foo):
    a = property(director_property.Foo.getA, director_property.Foo.setA)

    def ping(self):
        return "PyFoo::ping()"


foo = PyFoo()

foo.setA("BLABLA")
if foo.getA() != "BLABLA":
    raise RuntimeError

foo.a = "BIBI"
if foo.a != "BIBI":
    raise RuntimeError
