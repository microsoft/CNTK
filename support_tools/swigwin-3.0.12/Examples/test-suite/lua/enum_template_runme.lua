require("import")	-- the import fn
import("enum_template")	-- import lib
et=enum_template

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(et.eTest0 ~= nil)
assert(et.eTest1 ~= nil)

et.TakeETest(et.eTest0)

res = et.MakeETest()
et.TakeETest(res)
