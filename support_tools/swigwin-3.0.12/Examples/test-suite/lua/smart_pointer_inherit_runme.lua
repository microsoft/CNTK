require("import")	-- the import fn
import("smart_pointer_inherit")	-- import lib into global
spi=smart_pointer_inherit --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

der = spi.Derived(7)

ptr = spi.SmartDerived(der)

assert(ptr.val == 7)
assert(ptr:value() == 7)
assert(ptr:value2() == 7)
assert(ptr:value3() == 7)
assert(ptr:valuehide() == -1)
