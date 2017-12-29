require("import")	-- the import fn
import("equality")	-- import code
eq=equality    -- renaming import

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

-- === No equality operator ===

-- logically same data without equality operator are not equal
p1 = eq.MakePoint(10,9);
p2 = eq.MakePoint(10,9);

assert( p1 ~= p2 );

-- different wrappers for same Point* are equal
p3 = eq.GetZeroPointPtr()
p4 = eq.GetZeroPointPtr()

assert( p3 == p4 )


-- === Logically correct equality operator ===

ed1 = eq.EqualOpDefined(10)
ed2 = eq.EqualOpDefined(10)
ed3 = eq.EqualOpDefined(15)

assert( ed1 == ed2 )
assert( ed1 ~= ed3 )


-- === Logically incorrect equality operator ===

ew1 = eq.EqualOpWrong()
ew2 = eq.EqualOpWrong()

assert( ew1 ~= ew2 );

ew3 = eq.EqualOpWrong.GetStaticObject()
ew4 = eq.EqualOpWrong.GetStaticObject()

-- Even though these are pointers to same object, operator== overload should
-- state that they are not equal
assert( ew3 ~= ew4 )
