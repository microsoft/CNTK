require("import")	-- the import fn
import("disown")	-- import code

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

for x=0,100 do
    a=disown.A()
    b=disown.B()
    b:acquire(a)
end
collectgarbage() -- this will double delete unless the memory is managed properly

a=disown.A()
a:__disown()
b:remove(a)
a=nil
collectgarbage() -- this will double delete unless the manual disown call worked
