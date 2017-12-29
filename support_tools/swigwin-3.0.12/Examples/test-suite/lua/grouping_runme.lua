require("import")	-- the import fn
import("grouping")	-- import lib into global
g=grouping --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(g.test1(5) == 5)
g.test2(42) -- Return value is int* packed into userdata. We can't do anything with it

assert(g.test3 == 37)
g.test3 = 42
assert(g.test3 == 42)

assert(g.NEGATE ~= nil)
assert(g.do_unary(5, g.NEGATE) == -5)
