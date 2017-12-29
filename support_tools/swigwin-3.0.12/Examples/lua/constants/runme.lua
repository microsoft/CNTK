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

print("ICONST  = "..example.ICONST.." (should be 42)")
print("FCONST  = "..example.FCONST.." (should be 2.1828)")
print("CCONST  = "..example.CCONST.." (should be 'x')")
print("CCONST2 = "..example.CCONST2.." (this should be on a new line)")
print("SCONST  = "..example.SCONST.." (should be 'Hello World')")
print("SCONST2 = "..example.SCONST2.." (should be '\"Hello World\"')")
print("EXPR    = "..example.EXPR.." (should be 48.5484)")
print("iconst  = "..example.iconst.." (should be 37)")
print("fconst  = "..example.fconst.." (should be 3.14)")

-- helper to check that a fn failed
function checkfail(fn)
	if pcall(fn)==true then
		print("that shouldn't happen, it worked")
	else
		print("function failed as expected")
	end
end

-- these should fail
-- example.EXTERN is a nil value, so concatentatin will make it fail
checkfail(function() print("EXTERN = "..example.EXTERN) end)
checkfail(function() print("FOO = "..example.FOO) end)
