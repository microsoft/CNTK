require("import")	-- the import fn
import("extend")	-- import lib into global
e=extend --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

base1 = e.Base()
assert(base1.value == 0)
base2 = e.Base(10)
assert(base2.value == 10)

assert(base1:method(5) == 5)
assert(e.Base.zeroVal() == 0)
assert(base2:currentValue() == 10)
assert(base2:extendmethod(7) == 14)

der1 = e.Derived(0)
assert(der1.value == 0)
assert(der1:method(5) == 10)

der2 = e.Derived(17)
assert(der2.value == 34)
der2.extendval = 200.0
assert(math.abs(der2.actualval - 2.0) < 0.001)
assert(math.abs(der2.extendval - 200.0) < 0.001)
