require("import")	-- the import fn
import("nspace_extend")	-- import lib

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

ne = nspace_extend

-- Inner1

-- Constructors
in1_clr1 = ne.Outer.Inner1.Color()
in1_clr2 = ne.Outer.Inner1.Color.create()
in1_clr3 = ne.Outer.Inner1.Color(in1_clr2)

-- methods
in1_clr1:colorInstanceMethod(1.0)
ne.Outer.Inner1.Color.colorStaticMethod(2.0)

-- Inner2

-- Constructors
in2_clr1 = ne.Outer.Inner2.Color()
in2_clr2 = ne.Outer.Inner2.Color.create()
in2_clr3 = ne.Outer.Inner2.Color(in2_clr2)

assert(pcall(ne.Outer.Inner2.Color, in1_clr1) == false)

-- methods
in2_clr1:colorInstanceMethod(1.0)
ne.Outer.Inner2.Color.colorStaticMethod(2.0)

in2_clr3:colors(in1_clr1, in1_clr2, in2_clr2, in2_clr2, in2_clr3)

assert(pcall(in2_clr3.colors, in2_clr3,
  in2_clr1, in2_clr2, in1_clr2, in2_clr2, in2_clr3) == false)

