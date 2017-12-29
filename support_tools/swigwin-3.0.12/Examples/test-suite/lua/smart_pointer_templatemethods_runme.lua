require("import")	-- the import fn
import("smart_pointer_templatemethods")	-- import lib into global
spt=smart_pointer_templatemethods --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

o1 = spt.Objct()

iid = spt.InterfaceId()

po2 = o1:QueryInterfaceObjct(iid)
-- we can't call po2:DisposeObjct, because smart pointer Ptr<T> always return 0 when dereferencing
-- (see interface file). So we only check that po2 has necessary method
assert(po2.DisposeObjct ~= nil)
assert(po2.QueryInterfaceObjct ~= nil)
