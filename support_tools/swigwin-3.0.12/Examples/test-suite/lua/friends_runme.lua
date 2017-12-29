require("import")	-- the import fn
import("friends")	-- import lib into global
f=friends --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

f.globalscope()

b1 = f.B(5)
a1 = f.A(10)

assert(f.get_val1(a1) == 10)
assert(f.get_val1(a1, 2) == 12)
assert(f.get_val2(a1) == 20)
assert(f.get_val3(a1) == 30)

assert(f.get_val1(100, 1, 2) == 100)

assert(f.mix(a1,b1) == 15);

d1 = f.D_i(7)
assert(f.get_val1(d1) == 7)
f.set(d1,9)
assert(f.get_val1(d1) == 9)
