print "[lua] This is runme.lua"
-- test program for embedded lua
-- we do not need to load the library, as it was already in the interpreter
-- but let's check anyway
assert(type(example)=='table',"Don't appear to have loaded the example module")

-- a test function to run the tests
function do_tests()
	print("[lua] We are now in Lua, inside the do_tests() function")
	print("[lua] We will be calling example.gcd() and changing example.Foo")
	-- Call our gcd() function
	x = 42
	y = 105
	g = example.gcd(x,y)
	print("[lua] The gcd of",x,"and",y,"is",g)

	-- Manipulate the Foo global variable

	-- Output its current value
	print("[lua] Foo = ", example.Foo)

	-- Change its value
	example.Foo = 3.1415926

	-- See if the change took effect
	print("[lua] Foo = ", example.Foo)
	print("[lua] ending the do_tests() function")
end

function call_greeting()
	print("[lua] We are now in Lua, inside the call_greeting() function")
	example.greeting()
	print("[lua] ending the call_greeting() function")
end






