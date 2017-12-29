require("import")	-- the import fn
import("smart_pointer_simple")	-- import lib into global
sps=smart_pointer_simple --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

foo1 = sps.Foo()
foo1.x = 5
assert(foo1.x == 5)
assert(foo1:getx() == 5)

bar1 = sps.Bar(foo1)
bar1.x = 3
assert(bar1.x == 3)
assert(bar1:getx() == 3)

bar1.x = 5
assert(bar1.x == 5)
assert(bar1:getx() == 5)
