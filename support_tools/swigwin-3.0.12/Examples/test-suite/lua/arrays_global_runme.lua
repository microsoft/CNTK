require("import")	-- the import fn
import("arrays_global")	-- import lib
ag = arrays_global

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(ag.BeginString_FIX44a == "FIX.a.a")
assert(ag.BeginString_FIX44b == "FIX.b.b")

assert(ag.BeginString_FIX44c == "FIX.c.c")
assert(ag.BeginString_FIX44d == "FIX.d.d")
