---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

-- First create some objects using the pointer library.
print("Testing the pointer library")
a = example.new_intp()
b = example.new_intp()
c = example.new_intp()
example.intp_assign(a,37)
example.intp_assign(b,42)

print("     a = "..tostring(a))
print("     b = "..tostring(b))
print("     c = "..tostring(c))

-- Call the add() function with some pointers
example.add(a,b,c)

-- Now get the result
r = example.intp_value(c)
print("     37 + 42 = "..r)

-- Clean up the pointers
-- since this is C-style pointers you must clean it up
example.delete_intp(a)
example.delete_intp(b)
example.delete_intp(c)

-- Now try the typemap library
-- This should be much easier. Now how it is no longer
-- necessary to manufacture pointers.

print("Trying the typemap library")
r = example.sub(37,42)
print("     37 - 42 = "..r)

-- Now try the version with multiple return values

print("Testing multiple return values")
q,r = example.divide(42,37)
print("     42/37 = "..q.." remainder "..r)
