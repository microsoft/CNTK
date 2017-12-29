# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

swigexample

# Call some templated functions
swigexample.maxint(3,7)
swigexample.maxdouble(3.14,2.18)

# Create some class

iv = swigexample.vecint(100)
dv = swigexample.vecdouble(1000)

for i=0:99,
  iv.setitem(i,2*i);
end

for i=0:999,
  dv.setitem(i, 1.0/(i+1));
end;

sum = 0;
for i=0:99
      sum = sum + iv.getitem(i);
end
sum

sum = 0.0;
for i=0:999,
      sum = sum + dv.getitem(i);
end
sum

clear iv
clear dv
