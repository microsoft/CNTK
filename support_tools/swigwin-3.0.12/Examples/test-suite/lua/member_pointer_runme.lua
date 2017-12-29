--Example using pointers to member functions
require("import")	-- the import fn
import("member_pointer")	-- import code
mp = member_pointer

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

function check(what, expected, actual)
	assert(expected == actual,"Failed: "..what.." Expected: "..expected.." Actual: "..actual)
end

-- Get the pointers
area_pt = mp.areapt()
perim_pt = mp.perimeterpt()

-- Create some objects
s = mp.Square(10)

-- Do some calculations
check ("Square area ", 100.0, mp.do_op(s,area_pt))
check ("Square perim", 40.0, mp.do_op(s,perim_pt))

-- Try the variables
-- these have to still be part of the 'member_pointer' table
memberPtr = mp.areavar
memberPtr = mp.perimetervar

check ("Square area ", 100.0, mp.do_op(s,mp.areavar))
check ("Square perim", 40.0, mp.do_op(s,mp.perimetervar))

-- Modify one of the variables
mp.areavar = perim_pt

check ("Square perimeter", 40.0, mp.do_op(s,mp.areavar))

-- Try the constants
memberPtr = mp.AREAPT
memberPtr = mp.PERIMPT
memberPtr = mp.NULLPT

check ("Square area ", 100.0, mp.do_op(s,mp.AREAPT))
check ("Square perim", 40.0, mp.do_op(s,mp.PERIMPT))

