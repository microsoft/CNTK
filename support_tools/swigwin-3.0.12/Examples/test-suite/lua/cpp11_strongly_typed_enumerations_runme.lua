require("import")	-- the import fn
import("cpp11_strongly_typed_enumerations")	-- import lib

-- catch "undefined" global variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

function enumCheck(actual, expected)
  if not (actual == expected) then
    error("Enum value mismatch. Expected: "..expected.." Actual: "..actual)
  end
  return expected + 1
end

--val = 0
--val = enumCheck(cpp11_strongly_typed_enumerations.cpp11_strongly_typed_enumerations.Enum1_Val1, val)
local val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Enum1_Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum1_Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum1_Val3, 13)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum1_Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum1_Val5a, 13)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum1_Val6a, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Enum2_Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum2_Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum2_Val3, 23)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum2_Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum2_Val5b, 23)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum2_Val6b, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Val3, 43)
val = enumCheck(cpp11_strongly_typed_enumerations.Val4, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Enum5_Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum5_Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum5_Val3, 53)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum5_Val4, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Enum6_Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum6_Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum6_Val3, 63)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum6_Val4, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Enum7td_Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum7td_Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum7td_Val3, 73)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum7td_Val4, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Enum8_Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum8_Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum8_Val3, 83)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum8_Val4, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Enum10_Val1, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum10_Val2, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum10_Val3, 103)
val = enumCheck(cpp11_strongly_typed_enumerations.Enum10_Val4, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum12_Val1, 1121)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum12_Val2, 1122)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum12_Val3, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum12_Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum12_Val5c, 1121)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum12_Val6c, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Val1, 1131)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Val2, 1132)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Val3, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Val5d, 1131)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Val6d, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum14_Val1, 1141)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum14_Val2, 1142)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum14_Val3, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum14_Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum14_Val5e, 1141)
val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Enum14_Val6e, val)

-- Requires nested class support to work
--val = 0
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val1, 3121)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val2, 3122)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val3, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val4, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val5f, 3121)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val6f, val)
--
--val = 0
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Val1, 3131)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Val2, 3132)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Val3, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Val4, val)
--
--val = 0
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum14_Val1, 3141)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum14_Val2, 3142)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum14_Val3, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum14_Val4, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum14_Val5g, 3141)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum14_Val6g, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum12_Val1, 2121)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum12_Val2, 2122)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum12_Val3, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum12_Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum12_Val5h, 2121)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum12_Val6h, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Val1, 2131)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Val2, 2132)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Val3, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Val5i, 2131)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Val6i, val)

val = 0
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum14_Val1, 2141)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum14_Val2, 2142)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum14_Val3, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum14_Val4, val)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum14_Val5j, 2141)
val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Enum14_Val6j, val)

-- Requires nested class support to work
--val = 0
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum12_Val1, 4121)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum12_Val2, 4122)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum12_Val3, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum12_Val4, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum12_Val5k, 4121)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum12_Val6k, val)
--
--val = 0
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Val1, 4131)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Val2, 4132)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Val3, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Val4, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Val5l, 4131)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Val6l, val)
--
--val = 0
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum14_Val1, 4141)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum14_Val2, 4142)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum14_Val3, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum14_Val4, val)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum14_Val5m, 4141)
--val = enumCheck(cpp11_strongly_typed_enumerations.Class2.Struct1.Enum14_Val6m, val)

class1 = cpp11_strongly_typed_enumerations.Class1()
enumCheck(class1:class1Test1(cpp11_strongly_typed_enumerations.Enum1_Val5a), 13)
enumCheck(class1:class1Test2(cpp11_strongly_typed_enumerations.Class1.Enum12_Val5c), 1121)
--enumCheck(class1:class1Test3(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val5f), 3121)

enumCheck(cpp11_strongly_typed_enumerations.globalTest1(cpp11_strongly_typed_enumerations.Enum1_Val5a), 13)
enumCheck(cpp11_strongly_typed_enumerations.globalTest2(cpp11_strongly_typed_enumerations.Class1.Enum12_Val5c), 1121)
--enumCheck(globalTest3(cpp11_strongly_typed_enumerations.Class1.Struct1.Enum12_Val5f), 3121)

