require("import")	-- the import fn
import("li_typemaps")	-- import code

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

-- Check double INPUT typemaps
assert(li_typemaps.in_double(22.22) == 22.22)
assert(li_typemaps.inr_double(22.22) == 22.22)

-- Check double OUTPUT typemaps
assert(li_typemaps.out_double(22.22) == 22.22)
assert(li_typemaps.outr_double(22.22) == 22.22)

-- Check double INOUT typemaps
assert(li_typemaps.inout_double(22.22) == 22.22)
assert(li_typemaps.inoutr_double(22.22) == 22.22)

-- check long long
assert(li_typemaps.in_ulonglong(20)==20)
assert(li_typemaps.inr_ulonglong(20)==20)
assert(li_typemaps.out_ulonglong(20)==20)
assert(li_typemaps.outr_ulonglong(20)==20)
assert(li_typemaps.inout_ulonglong(20)==20)
assert(li_typemaps.inoutr_ulonglong(20)==20)

-- check bools
assert(li_typemaps.in_bool(true)==true)
assert(li_typemaps.inr_bool(false)==false)
assert(li_typemaps.out_bool(true)==true)
assert(li_typemaps.outr_bool(false)==false)
assert(li_typemaps.inout_bool(true)==true)
assert(li_typemaps.inoutr_bool(false)==false)

-- the others
a,b=li_typemaps.inoutr_int2(1,2)
assert(a==1 and b==2)

f,i=li_typemaps.out_foo(10)
assert(f.a==10 and i==20)
