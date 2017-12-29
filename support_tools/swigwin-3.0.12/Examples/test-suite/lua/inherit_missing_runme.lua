require("import")	-- the import fn
import("inherit_missing")	-- import lib
im=inherit_missing

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

bar = im.Bar()
spam = im.Spam()

assert(im.do_blah(bar) == "Bar::blah")
assert(im.do_blah(spam) == "Spam::blah")
