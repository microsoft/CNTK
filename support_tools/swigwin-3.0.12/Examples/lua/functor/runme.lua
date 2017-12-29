-- Operator overloading example
---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

a = example.intSum(0)
b = example.doubleSum(100.0)

-- Use the objects.  They should be callable just like a normal
-- lua function.

for i=0,100 do
    a(i)                -- Note: function call
    b(math.sqrt(i))     -- Note: function call
end 
print("int sum 0..100 is",a:result(),"(expected 5050)")
print("double sum 0..100 is",b:result(),"(expected ~771.46)")

