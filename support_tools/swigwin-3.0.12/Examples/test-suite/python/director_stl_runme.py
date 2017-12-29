import director_stl


class MyFoo(director_stl.Foo):

    def ping(self, s):
        return "MyFoo::ping():" + s

    def pident(self, arg):
        return arg

    def vident(self, v):
        return v

    def vidents(self, v):
        return v

    def vsecond(self, v1, v2):
        return v2


a = MyFoo()

a.tping("hello")
a.tpong("hello")

p = (1, 2)
a.pident(p)
v = (3, 4)
a.vident(v)

a.tpident(p)
a.tvident(v)

v1 = (3, 4)
v2 = (5, 6)
a.tvsecond(v1, v2)

vs = ("hi", "hello")
vs
a.tvidents(vs)
