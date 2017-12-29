-- demo of lua swig capacilities (operator overloading)
require("import")	-- the import fn
import("cpp_static")	-- import lib into global
cs=cpp_static --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

cs.StaticMemberTest.static_int = 5;
assert(cs.StaticMemberTest.static_int == 5);

cs.StaticFunctionTest.static_func()
cs.StaticFunctionTest.static_func_2(2)
cs.StaticFunctionTest.static_func_3(3,3)
