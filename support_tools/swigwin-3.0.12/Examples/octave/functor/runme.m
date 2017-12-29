# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

# Operator overloading example
swigexample

a = swigexample.intSum(0);
b = swigexample.doubleSum(100.0);

# Use the objects.  They should be callable just like a normal
# python function.

for i=0:100-1,
    a(i);           # Note: function call
    b(sqrt(i));     # Note: function call
endfor

a.result()
b.result()
