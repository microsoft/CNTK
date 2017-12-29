require("import")	-- the import fn
import("extend_placement")	-- import lib into global
ep=extend_placement --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

function test_obj(main, suppl)
  assert(main:spam() == 1)
  assert(main:spam("this_is_string") == 2)
  assert(main:spam(5) == 5)
  assert(main:spam(5,6) == 11)
  assert(main:spam(7,8,9) == 15)
  assert(main:spam(suppl,12.0) == 0)
  assert(main:spam(suppl) == 0)
end

foo1 = ep.Foo(0)
foo2 = ep.Foo(1,2)
foo3 = ep.Foo()
test_obj(foo1,foo2)


bar1 = ep.Bar()
bar2 = ep.Bar(5)
test_obj(bar1,bar2)

fti1 = ep.FooTi(0)
fti2 = ep.FooTi(1,2)
fti3 = ep.FooTi()
test_obj(fti1,foo1)

bti1 = ep.BarTi()
bti2 = ep.BarTi(5)
test_obj(bti1,bar1)
