import director_enum


class MyFoo(director_enum.Foo):

    def say_hi(self, val):
        return val


b = director_enum.Foo()
a = MyFoo()

if a.say_hi(director_enum.hello) != b.say_hello(director_enum.hi):
    raise RuntimeError
