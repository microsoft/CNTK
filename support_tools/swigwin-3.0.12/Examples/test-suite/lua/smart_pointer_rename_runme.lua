require("import")	-- the import fn
import("smart_pointer_rename")	-- import lib into global
spr=smart_pointer_rename --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})


foo = spr.Foo()
assert(foo:ftest1(1) == 1)
assert(foo:ftest2(1,2) == 2)

bar = spr.Bar(foo)
assert(bar:test() == 3)
assert(bar:ftest1(1) == 1)
assert(bar:ftest2(1,2) == 2)
