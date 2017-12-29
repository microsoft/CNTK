require("import")	-- the import fn
import("keyword_rename_c")	-- import lib into global
kn=keyword_rename_c--alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

-- Check renaming of Lua keywords
assert(kn.c_end(5) == 5)
assert(kn.c_nil(7) == 7)
