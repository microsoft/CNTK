-- file: example.lua

---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

-- throw a lot of exceptions:
-- you must catch exceptions using pcall and then checking the result

t = example.Test()

print "calling t:unknown()"
ok,res=pcall(function() t:unknown() end)
if ok then 
    print "  that worked! Funny"
else
    print("  call failed with error:",res)
end

print "calling t:simple()"
ok,res=pcall(function() t:simple() end)
if ok then 
    print "  that worked! Funny"
else
    print("  call failed with error:",res)
end

print "calling t:message()"
ok,res=pcall(function() t:message() end)
if ok then 
    print "  that worked! Funny"
else
    print("  call failed with error:",res)
end

print "calling t:hosed()"
ok,res=pcall(function() t:hosed() end)
if ok then 
    print "  that worked! Funny"
else
    print("  call failed with error:",res.code,res.msg)
end

-- this is a rather strange way to perform the multiple catch of exceptions
print "calling t:mutli()"
for i=1,3 do
    ok,res=pcall(function() t:multi(i) end)
    if ok then
        print "  that worked! Funny"
    else
        if swig_type(res)=="Exc *" then
            print("  call failed with Exc exception:",res.code,res.msg)
        else
            print("  call failed with error:",res)
        end
    end
end

-- this is a bit crazy, but it shows obtaining of the stacktrace
function a() 
    b() 
end
function b() 
    t:message()
end
print [[
Now let's call function a()
 which calls b() 
 which calls into C++
 which will throw an exception!]]
ok,res=pcall(a)
if ok then 
    print "  that worked! Funny"
else
    print("  call failed with error:",res)
end
print "Now let's do the same using xpcall(a,debug.traceback)"
ok,res=xpcall(a,debug.traceback)
if ok then 
    print "  that worked! Funny"
else
    print("  call failed with error:",res)
end
print "As you can see, the xpcall gives a nice stacktrace to work with"


ok,res=pcall(a)
print(ok,res)
ok,res=xpcall(a,debug.traceback)
print(ok,res)
