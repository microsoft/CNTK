
require("import")	-- the import fn
import("cpp_typedef")	-- import lib into global
ct = cpp_typedef --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

foo1 = ct.Foo()
bar1 = foo1:bar()
bar2 = ct.Foo.sbar()

u1 = ct.UnnamedStruct()
n1 = ct.TypedefNamedStruct()

test = ct.Test()

u2 = test:test1(u1)
n2 = test:test2(n1)
n3 = test:test3(n1)
n4 = test:test4(n1)
