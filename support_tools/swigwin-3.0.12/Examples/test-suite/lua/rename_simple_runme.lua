require("import")	-- the import fn
import("rename_simple")	-- import lib
rs = rename_simple

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(rs.NewStruct ~= nil)
assert(rs.NewStruct.NewStaticVariable == 444)
assert(rs.NewStruct_NewStaticVariable == 444)

assert(rs.NewStruct.NewStaticMethod() == 333)
assert(rs.NewStruct_NewStaticMethod() == 333)

assert(rs.NewStruct.ONE == 1)
assert(rs.NewStruct_ONE == 1)

assert(rs.NewFunction() == 555)

assert(rs.OldStruct == nil)
assert(rs.OldFunction == nil)
assert(rs.OldGlobalVariable == nil)
assert(rs.OldStruct_ONE == nil)
