# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

swigexample;

# First create some objects using the pointer library.
printf("Testing the pointer library\n");
a = swigexample.new_intp();
b = swigexample.new_intp();
c = swigexample.new_intp();
swigexample.intp_assign(a,37);
swigexample.intp_assign(b,42);

a,b,c

# Call the add() function with some pointers
swigexample.add(a,b,c);

# Now get the result
r = swigexample.intp_value(c);
printf("     37 + 42 = %i\n",r);

# Clean up the pointers
swigexample.delete_intp(a);
swigexample.delete_intp(b);
swigexample.delete_intp(c);

# Now try the typemap library
# This should be much easier. Now how it is no longer
# necessary to manufacture pointers.

printf("Trying the typemap library\n");
r = swigexample.sub(37,42);
printf("     37 - 42 = %i\n",r);

# Now try the version with multiple return values

printf("Testing multiple return values\n");
[q,r] = swigexample.divide(42,37);
printf("     42/37 = %d remainder %d\n",q,r);
