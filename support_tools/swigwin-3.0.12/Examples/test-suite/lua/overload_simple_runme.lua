require("import")	-- the import fn
import("overload_simple")	-- import code
for k,v in pairs(overload_simple) do _G[k]=v end -- move to global

-- lua has only one numeric type, foo(int) and foo(double) are the same
-- whichever one was wrapper first will be used

assert(foo(3)=="foo:int" or foo(3)=="foo:double") -- could be either
assert(foo("hello")=="foo:char *")

f=Foo()
b=Bar()

assert(foo(f)=="foo:Foo *")
assert(foo(b)=="foo:Bar *")

v = malloc_void(32)

assert(foo(v) == "foo:void *")

s = Spam()

assert(s:foo(3) == "foo:int" or s:foo(3.0) == "foo:double") -- could be either
assert(s:foo("hello") == "foo:char *")
assert(s:foo(f) == "foo:Foo *")
assert(s:foo(b) == "foo:Bar *")
assert(s:foo(v) == "foo:void *")

assert(Spam_bar(3) == "bar:int" or Spam_bar(3.0) == "bar:double")
assert(Spam_bar("hello") == "bar:char *")
assert(Spam_bar(f) == "bar:Foo *")
assert(Spam_bar(b) == "bar:Bar *")
assert(Spam_bar(v) == "bar:void *")

-- Test constructors

s = Spam()
assert(s.type == "none")

s = Spam(3)
assert(s.type == "int" or s.type == "double")

s = Spam("hello")
assert(s.type == "char *")

s = Spam(f)
assert(s.type == "Foo *")

s = Spam(b)
assert(s.type == "Bar *")

s = Spam(v)
assert(s.type == "void *")

free_void(v)
