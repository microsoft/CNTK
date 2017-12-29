require("import")	-- the import fn
import("enum_plus")	-- import lib
ep=enum_plus

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(ep.iFoo_Phoo == 50) -- Old variant of enum bindings
assert(ep.iFoo.Phoo == 50) -- New variant of enum bindings
