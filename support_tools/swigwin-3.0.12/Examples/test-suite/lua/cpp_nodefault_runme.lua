-- Run file
require("import")	-- the import fn
import("cpp_nodefault")	-- import lib into global
cn=cpp_nodefault --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

foo1 = cn.Foo(1,2)
foo1.a = 5
assert(foo1.a == 5)

foo2 = cn.create(1,2)

cn.consume(foo1,foo2)

bar1 = cn.Bar()
bar1:consume(cn.gvar, foo2)
foo3 = bar1:create(1,2)

foo3.a = 6
assert(foo3.a == 6)
