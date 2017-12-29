print "[lua] This is runme.lua"
-- test program for embedded lua
-- we do not need to load the library, as it was already in the interpreter
-- but let's check anyway
assert(type(example)=='table',"Don't appear to have loaded the example module")

-- note: we will copy the functions from example table into global
-- this will help us later
for k,v in pairs(example) do _G[k]=v end

-- our add function
-- we will be calling this from C
function add(a,b)
    print("[lua] this is function add(",a,b,")")
    c=a+b
    print("[lua] returning",c)
    return c
end

function append(a,b)
    print("[lua] this is function append(",a,b,")")
    c=a..b
    print("[lua] returning",c)
    return c
end


