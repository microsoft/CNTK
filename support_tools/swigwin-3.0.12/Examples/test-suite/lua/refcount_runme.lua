require("import")	-- the import fn
import("refcount")	-- import lib
r=refcount

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

a = r.A()
assert(a:ref_count() == 1)

b1 = r.B(a)
assert(a:ref_count() == 2)

b2 = r.B.create(a)
assert(a:ref_count() == 3)

b3 = b2:cloner()
assert(a:ref_count() == 4)

rca = b1:get_rca() -- RCPtr<A> is not wrapped
assert(a:ref_count() == 5)

b4 = r.global_create(a)
assert(a:ref_count() == 6)
