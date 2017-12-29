-- require is only available in Lua 5.1

if string.sub(_VERSION,1,7)=='Lua 5.1' then

        -- Initially the package should not be loaded
        assert(package.loaded["lua_no_module_global"] == nil)

        -- Load the module
        the_module = require "lua_no_module_global"

        -- require should return the module table
        assert(the_module.hi_mom ~= nil)
        assert(the_module.hi_mom() == "hi mom!")

        -- But it should not end up in the global table _G, subject to
        -- the -nomoduleglobal swig option.
        assert(_G["lua_no_module_global"] == nil)

        -- According to the Lua 5.1 reference manual, require should also
        -- store the module table into package.loaded["name"]
        assert(package.loaded["lua_no_module_global"] == the_module)
        assert(package.loaded["lua_no_module_global"].hi_mom() == "hi mom!")

end
