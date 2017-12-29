require("import")	-- the import fn
import("template_extend1")	-- import lib into global
te=template_extend1 --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

lb = te.lBaz()
assert(lb:foo() == "lBaz::foo")

db = te.dBaz()
assert(db:foo() == "dBaz::foo")
