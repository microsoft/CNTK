require("import")	-- the import fn
import("cpp_enum")	-- import code
ce=cpp_enum    -- renaming import

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

assert(ce.ENUM_ONE ~= nil)
assert(ce.ENUM_TWO ~= nil)

-- Enums inside classes
assert(ce.Foo.Hi == 0)
assert(ce.Foo.Hello == 1);
-- old-style bindings
assert(ce.Foo_Hi == 0)
assert(ce.Foo_Hello == 1);

assert(ce.Hi == 0)
assert(ce.Hello == 1)
