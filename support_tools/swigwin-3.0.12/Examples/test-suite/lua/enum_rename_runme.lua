require("import")	-- the import fn
import("enum_rename")	-- import lib
er=enum_rename

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(er.M_Jan ~= nil)
assert(er.May ~= nil)
