require("import")	-- the import fn
import("global_namespace")	-- import lib into global
gn=global_namespace --alias

-- catching undefined variables
local env = _ENV -- Lua 5.2
if not env then env = getfenv () end -- Lua 5.1
setmetatable(env, {__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

k1 = gn.Klass1()
k2 = gn.Klass2()
k3 = gn.Klass3()
k4 = gn.Klass4()
k5 = gn.Klass5()
k6 = gn.Klass6()
k7 = gn.Klass7()

gn.KlassMethods.methodA(k1,k2,k3,k4,k5,k6,k7)
gn.KlassMethods.methodB(k1,k2,k3,k4,k5,k6,k7)

k1 = gn.getKlass1A()
k2 = gn.getKlass2A()
k3 = gn.getKlass3A()
k4 = gn.getKlass4A()
k5 = gn.getKlass5A()
k6 = gn.getKlass6A()
k7 = gn.getKlass7A()

gn.KlassMethods.methodA(k1,k2,k3,k4,k5,k6,k7)
gn.KlassMethods.methodB(k1,k2,k3,k4,k5,k6,k7)

k1 = gn.getKlass1B()
k2 = gn.getKlass2B()
k3 = gn.getKlass3B()
k4 = gn.getKlass4B()
k5 = gn.getKlass5B()
k6 = gn.getKlass6B()
k7 = gn.getKlass7B()

gn.KlassMethods.methodA(k1,k2,k3,k4,k5,k6,k7)
gn.KlassMethods.methodB(k1,k2,k3,k4,k5,k6,k7)

x1 = gn.XYZ1()
x2 = gn.XYZ2()
x3 = gn.XYZ3()
x4 = gn.XYZ4()
x5 = gn.XYZ5()
x6 = gn.XYZ6()
x7 = gn.XYZ7()

gn.XYZMethods.methodA(x1,x2,x3,x4,x5,x6,x7)
gn.XYZMethods.methodB(x1,x2,x3,x4,x5,x6,x7)

gn.AnEnumMethods.methodA(gn.anenum1, gn.anenum2, gn.anenum3)
gn.AnEnumMethods.methodB(gn.anenum1, gn.anenum2, gn.anenum3)

gn.TheEnumMethods.methodA(gn.theenum1, gn.theenum2, gn.theenum3)
gn.TheEnumMethods.methodB(gn.theenum1, gn.theenum2, gn.theenum3)
