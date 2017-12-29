require("import")	-- the import fn
import("static_const_member")	-- import lib into global
scm=static_const_member --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(scm.X.PN == 0)
assert(scm.X.CN == 1)
assert(scm.X.EN == 2)
assert(scm.X.CHARTEST == "A")

-- Old-style bindings
assert(scm.X_PN == 0)
assert(scm.X_CN == 1)
assert(scm.X_EN == 2)
assert(scm.X_CHARTEST == "A")


