-- Operator overloading example
---- importing ----
if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	lib=loadlib('example.dll','luaopen_example') or loadlib('example.so','luaopen_example')
	assert(lib)()
else
	-- lua 5.1 does
	require('example')
end

print "ok, let's test Lua's ownership of C++ objects"
print("Currently there are",example.Shape_nshapes,"shapes (there should be 0)")

print "\nLet's make a couple"
a=example.Square(10)
b=example.Circle(1)
print("Currently there are",example.Shape_nshapes,"shapes (there should be 2)")

print "\nNote let's use the createX functions"
c=example.createCircle(5)
d=example.createSquare(3)
print("Currently there are",example.Shape_nshapes,"shapes (there should be 4)")

print "\nWe will run the garbage collector & see if they are till here"
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 4)")

print "\nLet's get rid of them all, collect garbage & see if they are till here"
a,b,c,d=nil,nil,nil,nil
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 0)")

print "\nLet's start putting stuff into the ShapeOwner"
print "The ShapeOwner now owns the shapes, but Lua still has pointers to them"
o=example.ShapeOwner()
a=example.Square(10)
b=example.Circle(1)
o:add(a)
o:add(b)
o:add(example.createSquare(5))
print("Currently there are",example.Shape_nshapes,"shapes (there should be 3)")

print "\nWe will nil our references,run the garbage collector & see if they are still here"
print "they should be, as the ShapeOwner owns them"
a,b=nil,nil
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 3)")

print "\nWe will access them and check that they are still valid"
a=o:get(0)
b=o:get(1)
print(" Area's are",a:area(),b:area(),o:get(2):area())
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 3)")

print "\nWe will remove one from the C++ owner & pass its ownership to Lua,"
print " then check that they are still unchanged"
a,b=nil,nil
a=o:remove(0) -- a now owns it
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 3)")

print "\nDelete the ShapeOwner (this should destroy two shapes),"
print " but we have one left in Lua"
o=nil
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 1)")

print "\nFinal tidy up "
a=nil
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 0)")


print "Final test, we will create some Shapes & pass them around like mad"
print "If there is any memory leak, you will see it in the memory usage"
io.flush()
sh={}
-- make some objects
for i=0,10 do
    a=example.Circle(i)
    b=example.Square(i)
    sh[a]=true
    sh[b]=true
end
o=example.ShapeOwner()
for i=0,10000 do
    for k,_ in pairs(sh) do
        o:add(k)
    end
    sh={}   -- clear it
    while true do
        a=o:remove(0)
        if a==nil then break end
        sh[a]=true
    end        
    if i%100==0 then collectgarbage() end
end
print "done"
o,sh=nil,nil
collectgarbage()
print("Currently there are",example.Shape_nshapes,"shapes (there should be 0)")
print "that's all folks!"
