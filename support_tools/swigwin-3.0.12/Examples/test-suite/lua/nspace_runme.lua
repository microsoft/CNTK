require("import")	-- the import fn
import("nspace")	-- import lib

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

ns = nspace

-- Inheritance
blue1 = ns.Outer.Inner3.Blue()

-- blue1:blueInstanceMethod()
blue1:colorInstanceMethod(60.0)
blue1.instanceMemberVariable = 4
assert( blue1.instanceMemberVariable == 4 )

-- Constructors
color1 = ns.Outer.Inner1.Color()
color2 = ns.Outer.Inner1.Color.create()
color = ns.Outer.Inner1.Color(color1)
color3 = ns.Outer.Inner2.Color.create()
color4 = ns.Outer.Inner2.Color.create()
color5 = ns.Outer.Inner2.Color.create()
mwp2 = ns.Outer.MyWorldPart2()
gc = ns.GlobalClass()

nnsp = ns.NoNSpacePlease()

-- Class methods
color:colorInstanceMethod(20.0)
ns.Outer.Inner1.Color.colorStaticMethod(30.0)
color3:colorInstanceMethod(40.0)
ns.Outer.Inner2.Color.colorStaticMethod(50.0)
color3:colors(color1, color2, color3, color4, color5)

gc:gmethod()

-- Class variables
color.instanceMemberVariable = 5
color1.instanceMemberVariable = 7
assert( color.instanceMemberVariable == 5 )
assert( color1.instanceMemberVariable == 7 )
assert(ns.Outer.Inner1.Color.staticMemberVariable == 0 )
assert(ns.Outer.Inner2.Color.staticMemberVariable == 0 )
ns.Outer.Inner1.Color.staticMemberVariable = 9
ns.Outer.Inner2.Color.staticMemberVariable = 11
assert(ns.Outer.Inner1.Color.staticMemberVariable == 9)
assert(ns.Outer.Inner2.Color.staticMemberVariable == 11)

-- Class constants
assert( ns.Outer.Inner1.Color.Specular == 0x20 )
assert( ns.Outer.Inner2.Color.Specular == 0x40 )
assert( ns.Outer.Inner1.Color.staticConstMemberVariable == 222 )
assert( ns.Outer.Inner2.Color.staticConstMemberVariable == 333 )
assert( ns.Outer.Inner1.Color.staticConstEnumMemberVariable ~= ns.Outer.Inner2.Color.staticConstEnumMemberVariable )


-- Aggregation
sc = ns.Outer.SomeClass()
assert( sc:GetInner1ColorChannel() ~= sc:GetInner2Channel() )
assert( sc:GetInner1Channel() ~= sc:GetInner2Channel() )

-- Backward compatibility 
assert(ns.Outer.Inner1.Diffuse ~= nil)
-- Enums within class within namespace shouldn't have backward compatible name. Same for static methods
assert(ns.Outer.Inner1.Color_Diffuse == nil)
assert(ns.Outer.Inner1.Color_colorStaticMethod == nil)

-- Enums and static methods of class marked as %nonspace should have backward compatible name
assert(ns.NoNSpacePlease_noNspaceStaticFunc() == 10)
assert(ns.Outer.Inner2.NoNSpacePlease_NoNspace == nil)
-- ReallyNoNSpaceEnum is wrapped into %nonspace and thus handled correctly.
-- NoNSpaceEnum is not (although both of them are in %nonspace-wrapped class) and thus
-- handled rather unexpectedly
assert(ns.NoNSpacePlease_ReallyNoNspace1 == 1)
assert(ns.NoNSpacePlease.ReallyNoNspace2 == 10)

