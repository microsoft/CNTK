-- file: runme.lua

-- This file illustrates class C++ interface generated
-- by SWIG.

---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

----- Object creation -----

print("Creating some objects:")
c = example.Circle(10)
print("    Created circle", c)
s = example.Square(10)
print("    Created square", s)

----- Access a static member -----

print("\nA total of",example.Shape_nshapes,"shapes were created")

----- Member data access -----

-- Set the location of the object

c.x = 20
c.y = 30

s.x = -10
s.y = 5

print("\nHere is their current position:")
print(string.format("    Circle = (%f, %f)",c.x,c.y))
print(string.format("    Square = (%f, %f)",s.x,s.y))

----- Call some methods -----

print("\nHere are some properties of the shapes:")
for _,o in pairs({c,s}) do
      print("   ", o)
      print("        area      = ", o:area())
      print("        perimeter = ", o:perimeter())
end

print("\nGuess I'll clean up now")

-- Note: this invokes the virtual destructor
c=nil
s=nil

-- call gc to make sure they are collected
collectgarbage()

print(example.Shape_nshapes,"shapes remain")
print "Goodbye"
