require("import")	-- the import fn
import("import_nomodule")	-- import code

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

f = import_nomodule.create_Foo()
import_nomodule.test1(f,42)
import_nomodule.delete_Foo(f)

b = import_nomodule.Bar()
import_nomodule.test1(b,37)

collectgarbage()