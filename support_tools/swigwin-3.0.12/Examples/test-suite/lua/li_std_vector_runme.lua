require("import")	-- the import fn
import("li_std_vector")	-- import code

for k,v in pairs(li_std_vector) do _G[k]=v end -- move to global

iv = IntVector(4)
for i=0,3 do
    iv[i] = i
end

for i=0,3 do assert(iv[i]==i) end

x = average(iv)

function near(x,y) return math.abs(x-y)<0.001 end

assert(near(x,1.5))

rv = RealVector()
rv:push_back(10)
rv:push_back(10.5)
rv:push_back(11)
rv:push_back(11.5)

a=half(rv)
for i=0,rv:size()-1 do
	assert(near(a[i],rv[i]/2))
end

dv = DoubleVector(10)
for i=0,9 do dv[i] = i/2.0 end

halve_in_place(dv)

for i=0,9 do 
	assert(near(dv[i],i/4))
end

sv=StructVector(4)

for i=0,3 do
	sv[i]=Struct(i)
end

for i=0,3 do
	assert(swig_type(sv[i]) =='Struct *' and sv[i].num==i)
end

-- range checking
idx=0
function test_set() iv[idx]=0 end
function test_get() iv[idx]=0 end

idx=0 --ok
assert(pcall(test_get)==true)
assert(pcall(test_set)==true)
idx=-1 --should error
assert(pcall(test_get)==false)
assert(pcall(test_set)==false)
idx=3 --ok
assert(pcall(test_get)==true)
assert(pcall(test_set)==true)
idx=4 --should error
assert(pcall(test_get)==false)
assert(pcall(test_set)==false)

