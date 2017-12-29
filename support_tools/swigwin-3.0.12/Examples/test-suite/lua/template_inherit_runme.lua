require("import")	-- the import fn
import("template_inherit")	-- import lib into global
ti=template_inherit --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})


fi = ti.FooInt()
assert(fi:blah() == "Foo")
assert(fi:foomethod() == "foomethod")

bi = ti.BarInt()
assert(bi:blah() == "Bar")
assert(bi:foomethod() == "foomethod")

assert(ti.invoke_blah_int(fi) == "Foo")
assert(ti.invoke_blah_int(bi) == "Bar")

bd = ti.BarDouble()
success = pcall(ti.invoke_blah_int, bd)
assert(not success)
