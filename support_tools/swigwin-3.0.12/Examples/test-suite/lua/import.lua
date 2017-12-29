-- import
-- the lua 5.0 loading mechanism is rather poor & relies upon the loadlib() fn
-- the lua 5.1 loading mechanism is simplicity itself
-- for now we need a bridge which will use the correct version

function import_5_0(name)
	-- imports the file into the program
	-- for a module 'example'
	-- this must load 'example.dll' or 'example.so'
	-- and look for the fn 'luaopen_example()'
	if rawget(_G,name)~=nil then return end -- module appears to be loaded
		
	local lib=loadlib(name..'.dll','luaopen_'..name) or loadlib(name..'.so','luaopen_'..name)
	assert(lib,"error loading module:"..name)
	
	lib() -- execute the function: initialising the lib
	assert(rawget(_G,name)~=nil,"no module table found")
end

function import_5_1(name)
	require(name)
end

if string.sub(_VERSION,1,7)=='Lua 5.0' then
	import=import_5_0
else
	import=import_5_1
end
