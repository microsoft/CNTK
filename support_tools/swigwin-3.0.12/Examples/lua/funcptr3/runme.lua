---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

a = 37
b = 42

-- Now call our C function

print("Trying some C functions")
print("    a        =", a)
print("    b        =", b)
print("    add(a,b) =", example.add(a,b))
print("    sub(a,b) =", example.sub(a,b))
print("    mul(a,b) =", example.mul(a,b))

print("Calling them using the my_func()")
print("    add(a,b) =", example.callback(a,b,example.add))
print("    sub(a,b) =", example.callback(a,b,example.sub))
print("    mul(a,b) =", example.callback(a,b,example.mul))

print("Now let us write our own function")
function foo(a,b) return 101 end
print("    foo(a,b) =", example.callback(a,b,foo))

print("Now let us try something that will fail")
local ok,c=pcall(example.callback,a,b,print)
if ok==false then
	print("this failed as expected, error:",c)
else
	print("oops, that worked! result:",c)
end


-- part2 stored function
print("trying a stored fn")
print("the_func=",example.the_func)
print("setting to print")
example.the_func=print
print("the_func=",example.the_func)
print("call_the_func(5)")
example.call_the_func(5)

function bar(i) print("bar",i) end
print("setting to bar")
example.the_func=bar
print("call_the_func(5)")
example.call_the_func(5)
