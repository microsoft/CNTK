require("import")	-- the import fn
import("global_vars")	-- import lib
gv = global_vars

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

gv.b = "abcde"
assert(gv.b == "abcde")

gv.a.x = 7
assert(gv.a.x == 7)

a1 = gv.A()
a1.x = 11
gv.a = a1
assert(gv.a.x == 11)

gv.x = 10
assert(gv.x == 10)

assert(gv.Hi ~= nil)
assert(gv.Hola ~= nil)

gv.h = gv.Hi
assert(gv.h == gv.Hi)


-- It is not clear whether these tests should work or not
-- Currently they don't.
--
-- assert(gv.c_member == 10)
--
-- gv.c_member = 5
-- assert(gv.x == 5)
--
-- gv.h = gv.Hi
-- assert(gv.hr == gv.Hi)
--
-- gv.hr = gv.Hola
-- assert(gv.h == gv.Hola)
-- assert(gv.hr == gv.Hola)
