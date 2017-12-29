require("import")	-- the import fn
import("smart_pointer_not")	-- import lib into global
spn=smart_pointer_not --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})


foo = spn.Foo()
foo.x = 7
assert(foo:getx() == 7)

bar = spn.Bar(foo)
success = pcall(bar.getx, bar) -- Bar is not a smart pointer. Call should fail
assert(not success)

spam = spn.Spam(foo)
success = pcall(spam.getx, spam) -- Spam is not a smart pointer. Call should fail
assert(not success)

grok = spn.Grok(foo)
success = pcall(grok.getx, grok) -- Spam is not a smart pointer. Call should fail
assert(not success)
