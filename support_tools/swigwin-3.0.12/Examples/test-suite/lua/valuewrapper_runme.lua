require("import")	-- the import fn
import("valuewrapper")	-- import code
v=valuewrapper    -- renaming import

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(v.Xi ~= nil)
assert(v.YXi ~= nil)

x1 = v.Xi(5)

y1 =v.YXi()
assert(y1:spam(x1) == 0)
assert(y1:spam() == 0)
