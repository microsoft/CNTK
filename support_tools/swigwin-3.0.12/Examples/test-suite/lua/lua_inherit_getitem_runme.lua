require("import")	-- the import fn
import("lua_inherit_getitem")	-- import lib

local t = lua_inherit_getitem;
local base = t.CBase()
local derived = t.CDerived()

assert(base.Foo ~= nil)
assert(base:Foo() == "CBase::Foo")
assert(derived.Foo == base.Foo)
assert(derived:Foo() == "CBase::Foo")

