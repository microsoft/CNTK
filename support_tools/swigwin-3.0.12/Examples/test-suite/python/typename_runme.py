import typename
import types
f = typename.Foo()
b = typename.Bar()

x = typename.twoFoo(f)
if not isinstance(x, types.FloatType):
    raise RuntimeError, "Wrong return type (FloatType) !"
y = typename.twoBar(b)
if not isinstance(y, types.IntType):
    raise RuntimeError, "Wrong return type (IntType)!"
