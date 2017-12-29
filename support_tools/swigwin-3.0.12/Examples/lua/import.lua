-- import
-- the lua 5.0 loading mechanism is rather poor & relies upon the loadlib() fn
-- the lua 5.1 loading mechanism is simplicity itself
-- for now we need a bridge which will use the correct version

function import_5_0(module)
	-- imports the file into the program
	-- for a module 'example'
	-- this must load 'example.dll' or 'example.so'
	-- and look for the fn 'Example_Init()' (note the capitalisation)
	if rawget(_G,module)~=nil then return end -- module appears to be loaded
		
	-- capitalising the first letter
	local c=string.upper(string.sub(module,1,1))
	local fnname=c..string.sub(module,2).."_Init"
	
	local suffix,lib
	-- note: as there seems to be no way in lua to determine the platform
	-- we will try loading all possible names
	-- providing one works, we can load
	for _,suffix in pairs{".dll",".so"} do
		lib=loadlib(module..suffix,fnname)
		if lib then -- found
			break
		end
	end
	assert(lib,"error loading module:"..module)
	
	lib() -- execute the function: initialising the lib
	local m=rawget(_G,module)	-- gets the module object
	assert(m~=nil,"no module table found")
end

function import_5_1(module)
	require(module)
end

if string.sub(_VERSION,1,7)=='Lua 5.0' then
	import=import_5_0
else
	import=import_5_1
end
