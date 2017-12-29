---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

-- Try to set the values of some global variables

example.ivar   =  42
example.svar   = -31000
example.lvar   =  65537
example.uivar  =  123456
example.usvar  =  61000
example.ulvar  =  654321
example.scvar  =  -13
example.ucvar  =  251
example.cvar   =  "S"
example.fvar   =  3.14159
example.dvar   =  2.1828
example.strvar =  "Hello World"
example.iptrvar= example.new_int(37)
example.ptptr  = example.new_Point(37,42)
example.name   = "Bill"

-- Now print out the values of the variables

print("Variables (values printed from Lua)")

print("ivar      =", example.ivar)
print("svar      =", example.svar)
print("lvar      =", example.lvar)
print("uivar     =", example.uivar)
print("usvar     =", example.usvar)
print("ulvar     =", example.ulvar)
print("scvar     =", example.scvar)
print("ucvar     =", example.ucvar)
print("fvar      =", example.fvar)
print("dvar      =", example.dvar)
print("cvar      =", example.cvar)
print("strvar    =", example.strvar)
print("cstrvar   =", example.cstrvar)
print("iptrvar   =", example.iptrvar)
print("name      =", example.name)
print("ptptr     =", example.ptptr, example.Point_print(example.ptptr))
print("pt        =", example.pt, example.Point_print(example.pt))

print("\nVariables (values printed from C)")

example.print_vars()

print "\nNow I'm going to try and modify some read only variables";

print "     Tring to set 'path' to 'Whoa!'";
if pcall(function() example.path = "Whoa!" end)==true then
	print "     Thats funny, it didn't give an error!"
else
	print "     It gave an error, as it should"
end
print("     Just checking the value: path      =", example.path)

print "     Trying to set 'status' to '0'";
if pcall(function() example.status = 0 end)==true then
	print "     Thats funny, it didn't give an error!"
else
	print "     It gave an error, as it should"
end
print("     Just checking the value: status    =", example.status)


print "\nI'm going to try and update a structure variable.\n"

example.pt = example.ptptr

print "The new value is"
example.pt_print()
print("You should see the value", example.Point_print(example.ptptr))

