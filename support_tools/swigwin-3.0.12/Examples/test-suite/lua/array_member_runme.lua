require("import")	-- the import fn
import("array_member")	-- import lib
am = array_member

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(am.get_value(am.global_data,0) == 0)
assert(am.get_value(am.global_data,7) == 7)

foo = am.Foo()
foo.data = am.global_data
assert(am.get_value(foo.data,0) == 0)

for i = 0, 7 do
  assert(am.get_value(foo.data,i) == am.get_value(am.global_data,i))
end


for i = 0, 7 do
  am.set_value(am.global_data,i,-i)
end

am.global_data = foo.data

for i = 0, 7 do
  assert(am.get_value(foo.data,i) == am.get_value(am.global_data,i))
end
