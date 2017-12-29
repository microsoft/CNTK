require("import")	-- the import fn
import("enum_scope_template")	-- import lib
est=enum_scope_template

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(est.TreeInt.Oak ~= nil)
assert(est.TreeInt_Oak ~= nil)
assert(est.TreeInt.Cedar ~= nil)
