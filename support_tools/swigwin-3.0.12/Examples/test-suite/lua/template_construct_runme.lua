require("import")	-- the import fn
import("template_construct")	-- import lib into global
tc=template_construct --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

foo = tc.Foo_int(1)
