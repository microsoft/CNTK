-- demo of lua swig capacilities (operator overloading)
require("import")	-- the import fn
import("exception_order")	-- import lib into global
eo=exception_order --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

a = eo.A()

function try1()
	a:foo()
end

function try2()
	a:bar()
end

function try3()
	a:foobar()
end

-- the following code used to work
-- but now no longer works, as the lua bindings don't throw objects any more
-- all objects are converted to string & thrown
-- it could be made to work, if E1 & E2 were thrown by value (see lua.swg)
--[[
ok,ex=pcall(try1)
print(ok,ex)
assert(ok==false and swig_type(ex)==swig_type(eo.E1()))

ok,ex=pcall(try2)
assert(ok==false and swig_type(ex)==swig_type(eo.E2()))
]]
-- this new code does work, but has to look at the string
ok,ex=pcall(try1)
assert(ok==false and ex=="object exception:E1")

ok,ex=pcall(try2)
assert(ok==false and ex=="object exception:E2")

-- the SWIG_exception is just an error string
ok,ex=pcall(try3)
assert(ok==false and type(ex)=="string")

