require("import")	-- the import fn
import("varargs")	-- import lib
v=varargs

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(v.test("Hello") == "Hello")
assert(v.test_def("Hello",0) == "Hello")

assert(v.Foo.statictest("Hello") == "Hello")
assert(v.Foo.statictest("Hello",1) == "Hello")

assert(v.test_plenty("Hello") == "Hello")
assert(v.test_plenty("Hello",1) == "Hello")
assert(v.test_plenty("Hello",1,2) == "Hello")
