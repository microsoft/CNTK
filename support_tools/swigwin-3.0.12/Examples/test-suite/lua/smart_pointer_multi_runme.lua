require("import")	-- the import fn
import("smart_pointer_multi")	-- import lib into global
spm=smart_pointer_multi --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

foo = spm.Foo()
foo.x = 5
assert(foo:getx() == 5)

bar = spm.Bar(foo)
spam = spm.Spam(bar)
grok = spm.Grok(bar)

assert(bar:getx() == 5)
assert(spam:getx() == 5)
spam.x = 7
assert(grok:getx() == 7)
grok.x = 10
assert(foo:getx() == 10)
