---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

-- Call our gcd() function
x = 42
y = 105
g = example.gcd(x,y)
print("The gcd of",x,"and",y,"is",g)

-- Manipulate the Foo global variable

-- Output its current value
print("Foo = ", example.Foo)

-- Change its value
example.Foo = 3.1415926

-- See if the change took effect
print("Foo = ", example.Foo)









