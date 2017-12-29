require("import")	-- the import fn
import("overload_complicated")	-- import lib into global
oc=overload_complicated --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(oc.foo(1,1,"test",1) == 15)

p1 = oc.Pop(nil)
p1 = oc.Pop(nil,false)

assert(p1:hip(true) == 701)
assert(p1:hip(nil) == 702)

assert(p1:hop(true) == 801)
assert(p1:hop(nil) == 805)

assert(oc.muzak(true) == 3001)
assert(oc.muzak(nil) == 3002)
