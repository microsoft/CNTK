require("import")	-- the import fn
import("funcptr_cpp")	-- import lib into global
fc=funcptr_cpp --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(fc.addByValue(5,10) == 15)
-- These two won't work. Lua will successfully store the answer as userdata, but there is
-- no way of accessing the insides of userdata.
-- assert(fc.addByPointer(7, 9) == 16)
-- assert(fc.addByReference(8, 9) == 17)

assert(fc.call1(fc.ADD_BY_VALUE, 5, 10) == 15)
assert(fc.call2(fc.ADD_BY_POINTER, 7, 9) == 16)
assert(fc.call3(fc.ADD_BY_REFERENCE, 8, 9) == 17)
assert(fc.call1(fc.ADD_BY_VALUE_C, 2, 3) == 5)
