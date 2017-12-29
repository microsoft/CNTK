require("import")	-- the import fn
import("extend_constructor_destructor")	-- import lib into global
ecd=extend_constructor_destructor --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

a1 = ecd.AStruct(101)
assert(a1.ivar == 101)
assert(ecd.globalVar == 101)

b1 = ecd.BStruct(201)
assert(b1.ivar == 201)
assert(ecd.globalVar == 201)

c1 = ecd.CStruct(301)
assert(c1.ivar == 301)
assert(ecd.globalVar == 301)

d1 = ecd.DStruct(401)
assert(d1.ivar == 401)
assert(ecd.globalVar == 401)

e1 = ecd.EStruct(501)
assert(e1.ivar == 501)
assert(ecd.globalVar == 501)

f1 = ecd.FStruct(601)
assert(f1.ivar == 601)
assert(ecd.globalVar == 601)
