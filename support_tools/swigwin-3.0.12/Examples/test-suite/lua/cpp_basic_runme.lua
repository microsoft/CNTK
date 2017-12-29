require("import")	-- the import fn
import("cpp_basic")	-- import code
cb=cpp_basic    -- renaming import

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

f=cb.Foo(4)
assert(f.num==4)
f.num=-17
assert(f.num==-17)

b=cb.Bar()

b.fptr=f
assert(b.fptr.num==-17)
assert(b:test(-3,b.fptr)==-5)
f.num=12
assert(b.fptr.num==12)

assert(b.fref.num==-4)
assert(b:test(12,b.fref)==23)

-- references don't take ownership, so if we didn't define this here it might get garbage collected
f2=cb.Foo(23)
b.fref=f2
assert(b.fref.num==23)
assert(b:test(-3,b.fref)==35)

assert(b.fval.num==15)
assert(b:test(3,b.fval)==33)
b.fval=cb.Foo(-15)  -- this is safe as it is copied into the C++
assert(b.fval.num==-15)
assert(b:test(3,b.fval)==-27)

f3=b:testFoo(12,b.fref)
assert(f3.num==32)

-- now test global
f4=cb.Foo(6)
cb.Bar_global_fptr=f4
assert(cb.Bar_global_fptr.num==6)
assert(cb.Bar.global_fptr.num==6)
f4.num=8
assert(cb.Bar_global_fptr.num==8)
assert(cb.Bar.global_fptr.num==8)

assert(cb.Bar_global_fref.num==23)
assert(cb.Bar.global_fref.num==23)
cb.Bar_global_fref=cb.Foo(-7) -- this will set the value
assert(cb.Bar_global_fref.num==-7)
assert(cb.Bar.global_fref.num==-7)

assert(cb.Bar_global_fval.num==3)
assert(cb.Bar.global_fval.num==3)
cb.Bar_global_fval=cb.Foo(-34)
assert(cb.Bar_global_fval.num==-34)
assert(cb.Bar.global_fval.num==-34)

assert(cb.Bar.global_cint == -4)
assert(cb.Bar_global_cint == -4)

-- Now test member function pointers
func1_ptr=cb.get_func1_ptr()
func2_ptr=cb.get_func2_ptr()
f.num=4
assert(f:func1(2)==16)
assert(f:func2(2)==-8)

f.func_ptr=func1_ptr
assert(cb.test_func_ptr(f,2)==16)
f.func_ptr=func2_ptr
assert(cb.test_func_ptr(f,2)==-8)

-- Test that __tostring metamethod produce no internal asserts
f2_name = tostring(f2)

f3 = cb.FooSub()
f3_name = tostring(f3)

f4 = cb.FooSubSub()
f4_name = tostring(f4)

assert( f2_name == "Foo" )
assert( f3_name == "Foo" )
assert( f4_name == "FooSubSub" )

-- Test __eq implementation supplied by default

-- eq_f1 and eq_f2 must be different userdata with same Foo* pointer. If eq_f1 and eq_f2 are the same userdata (e.g.)
-- > eq_f1 = smth
-- > eq_f2 = eq_f1
-- then default Lua equality comparison kicks in and considers them equal. Access to global_fptr is actually a
-- function call (internally) and it returns new userdata each time.
eq_f1 = cb.Bar.global_fptr
eq_f2 = cb.Bar.global_fptr
assert( eq_f1 == eq_f2 )
