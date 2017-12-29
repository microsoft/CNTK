---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end


x,y = 42,105
print("add1 - simple arg passing and single return value -- ")
print(example.add1(x,y))
print("add2 - pointer arg passing and single return value through pointer arg -- ")
print(example.add2(x,y))
print("add3 - simple arg passing and double return value through return and ptr arg -- ")
print(example.add3(x,y))
print("add4 - dual use arg and return value -- ")
print(example.add4(x,y))
