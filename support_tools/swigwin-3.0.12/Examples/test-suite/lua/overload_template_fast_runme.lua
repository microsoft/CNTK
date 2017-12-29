require("import")	-- the import fn
import("overload_template_fast")	-- import code
for k,v in pairs(overload_template_fast) do _G[k]=v end -- move to global

-- lua has only one numeric type, so maximum(int,int) and maximum(double,double) are the same
-- whichever one was wrapper first will be used (which is int)

f = foo()

a = maximum(3,4)

-- mix 1
assert(mix1("hi") == 101)
assert(mix1(1.0, 1.0) == 102)
assert(mix1(1.0) == 103)

-- mix 2
assert(mix2("hi") == 101)
assert(mix2(1.0, 1.0) == 102)
assert(mix2(1.0) == 103)

-- mix 3
assert(mix3("hi") == 101)
assert(mix3(1.0, 1.0) == 102)
assert(mix3(1.0) == 103)

-- Combination 1
assert(overtparams1(100) == 10)
assert(overtparams1(100.0, 100) == 20)

-- Combination 2
assert(overtparams2(100.0, 100) == 40)

-- Combination 3
assert(overloaded() == 60)
assert(overloaded(100.0, 100) == 70)

-- Combination 4
assert(overloadedagain("hello") == 80)
assert(overloadedagain() == 90)

-- specializations
assert(specialization(10) == 202 or specialization(10.0) == 203) -- only one works
assert(specialization(10, 10) == 204 or specialization(10.0, 10.0) == 205) -- ditto
assert(specialization("hi", "hi") == 201)

-- simple specialization
xyz()
xyz_int()
xyz_double()

-- a bit of everything
assert(overload("hi") == 0)
assert(overload(1) == 10)
assert(overload(1, 1) == 20)
assert(overload(1, "hello") == 30)

k = Klass()
assert(overload(k) == 10)
assert(overload(k, k) == 20)
assert(overload(k, "hello") == 30)
-- this one is incorrect: it mactches overload(10.0, "hi") with int overload(T t, const char *c)
--print(overload(10.0, "hi"))
--assert(overload(10.0, "hi") == 40)
assert(overload() == 50)

-- everything put in a namespace
assert(nsoverload("hi") == 1000,"nsoverload()")
assert(nsoverload(1) == 1010,"nsoverload(int t)")
assert(nsoverload(1, 1) == 1020,"nsoverload(int t, const int &)")
assert(nsoverload(1, "hello") == 1030,"nsoverload(int t, const char *)")
assert(nsoverload(k) == 1010,"nsoverload(Klass t)")
assert(nsoverload(k, k) == 1020,"nsoverload(Klass t, const Klass &)")
assert(nsoverload(k, "hello") == 1030,"nsoverload(Klass t, const char *)")
-- again this one fails
--assert(nsoverload(10.0, "hi") == 1040,"nsoverload(double t, const char *)")
assert(nsoverload() == 1050,"nsoverload(const char *)")

A_foo(1)
b = B()
b:foo(1)
