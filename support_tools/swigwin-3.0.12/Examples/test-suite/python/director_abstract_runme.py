import director_abstract


def is_new_style_class(cls):
    return hasattr(cls, "__class__")


class MyFoo(director_abstract.Foo):

    def __init__(self):
        director_abstract.Foo.__init__(self)

    def ping(self):
        return "MyFoo::ping()"


a = MyFoo()

if a.ping() != "MyFoo::ping()":
    raise RuntimeError, a.ping()

if a.pong() != "Foo::pong();MyFoo::ping()":
    raise RuntimeError, a.pong()


class MyExample1(director_abstract.Example1):

    def Color(self, r, g, b):
        return r


class MyExample2(director_abstract.Example2):

    def Color(self, r, g, b):
        return g


class MyExample3(director_abstract.Example3_i):

    def Color(self, r, g, b):
        return b

me1 = MyExample1()
if director_abstract.Example1_get_color(me1, 1, 2, 3) != 1:
    raise RuntimeError

if is_new_style_class(MyExample2):
    MyExample2_static = MyExample2
else:
    MyExample2_static = MyExample2(0, 0)
me2 = MyExample2(1, 2)
if MyExample2_static.get_color(me2, 1, 2, 3) != 2:
    raise RuntimeError

if is_new_style_class(MyExample3):
    MyExample3_static = MyExample3
else:
    MyExample3_static = MyExample3()
me3 = MyExample3()
if MyExample3_static.get_color(me3, 1, 2, 3) != 3:
    raise RuntimeError

error = 1
try:
    me1 = director_abstract.Example1()
except:
    error = 0
if (error):
    raise RuntimeError

error = 1
try:
    me2 = director_abstract.Example2()
except:
    error = 0
if (error):
    raise RuntimeError

error = 1
try:
    me3 = director_abstract.Example3_i()
except:
    error = 0
if (error):
    raise RuntimeError


try:
    f = director_abstract.A.f
except:
    raise RuntimeError
