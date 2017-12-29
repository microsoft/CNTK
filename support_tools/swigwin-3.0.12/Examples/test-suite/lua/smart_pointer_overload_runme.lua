require("import")	-- the import fn
import("smart_pointer_overload")	-- import code
for k,v in pairs(smart_pointer_overload) do _G[k]=v end -- move to global

f = Foo()
b = Bar(f)

assert(f:test(3) == 1)
--assert(f:test(3.5) == 2)	-- won't work due to being unable to overloads
assert(f:test("hello") == 3)

assert(b:test(3) == 1)
--assert(b:test(3.5) == 2)	-- won't work due to being unable to overloads
assert(b:test("hello") == 3)
