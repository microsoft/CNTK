require("import")	-- the import fn
import("iadd")	-- import lib into global
i=iadd --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

foo1 = i.Foo()
foo1_a = foo1.AsA
assert(foo1_a.x == 5)
assert(foo1_a:get_me().x == 5)
-- Unfortunately, in Lua operator+= can't be overloaded

foo1.AsLong = 1000
assert(foo1.AsLong == 1000)
