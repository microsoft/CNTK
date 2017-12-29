require("import")	-- the import fn
import("nested_workaround")	-- import lib
nw=nested_workaround

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

i1 = nw.Inner(5)
assert(i1:getValue() == 5)
i1:setValue(7)
assert(i1:getValue() == 7)

o1 = nw.Outer()
i2 = o1:createInner(9)
assert(i2:getValue() == 9)
i2:setValue(11)
assert(o1:getInnerValue(i2) == 11)

i3 = o1:doubleInnerValue(i2)
assert(i3:getValue() == 22)
