require("import")	-- the import fn
import("extend_variable")	-- import lib into global
ev=extend_variable --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

e1 = ev.ExtendMe()
answ = 1.0
assert(e1:set(7.0))
--assert(e1:get(answ)) -- doesn't work. Lua can't pass primitive type by non-const reference
--assert(answ == 7.0)

e1.ExtendVar = 5.0
assert(e1.ExtendVar == 5.0)

assert(ev.Foo.Bar == 42)
assert(ev.Foo.AllBarOne == 4422)

assert(ev.Foo.StaticInt == 1111)
ev.Foo.StaticInt = 3333
assert(ev.Foo.StaticInt == 3333)

assert(ev.Foo.StaticConstInt == 2222)

b1 = ev.Bar()
assert(b1.x == 1)
