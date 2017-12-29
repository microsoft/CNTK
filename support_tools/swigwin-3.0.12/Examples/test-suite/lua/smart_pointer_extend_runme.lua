require("import")	-- the import fn
import("smart_pointer_extend")	-- import lib into global
spe=smart_pointer_extend --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(spe.CBase.hello() == 1)
assert(spe.CBase.z == 1)

base1 = spe.CBase()
base1.x = 7

p1 = spe.CPtr()

assert(spe.get_hello(p1) == 1)
assert(p1:foo() == 1)
assert(p1:bar() == 2)
assert(p1:boo(5) == 5)

foo = spe.Foo()
bar = spe.Bar(foo)

assert(bar:extension(5,7) == 5)
assert(bar:extension(7) == 7)
assert(bar:extension() == 1)

dfoo = spe.DFoo()
dptr = spe.DPtrFoo(dfoo)

assert(dptr:Ext() == 2)
assert(dptr:Ext(5) == 5)
