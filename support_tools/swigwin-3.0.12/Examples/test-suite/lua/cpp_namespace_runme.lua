require("import")	-- the import fn
import("cpp_namespace")	-- import lib into global
cn=cpp_namespace --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(cn.fact(4) == 24)
assert(cn.Foo == 42)

t1 = cn.Test()
assert(t1:method() == "Test::method")

cn.weird("t1", 4)

assert(cn.do_method(t1) == "Test::method")
assert(cn.do_method2(t1) == "Test::method")

t2 = cn.Test2()
assert(t2:method() == "Test2::method")


assert(cn.foo3(5) == 5)

assert(cn.do_method3(t2, 7) == "Test2::method")
