require("import")	-- the import fn
import("fvirtual")	-- import lib into global
f=fvirtual --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

n1 = f.Node()
n2 = f.Node()
assert(n1:addChild(n2) == 1)

ns = f.NodeSwitch()
assert(ns:addChild(n2) == 2)
assert(ns:addChild(ns) == 2)
assert(ns:addChild(n1, false) == 3)
