# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

swigexample

a = 37
b = 42

# Now call our C function with a bunch of callbacks

printf("Trying some C callback functions\n");
printf("    a        = %i\n", a);
printf("    b        = %i\n", b);
printf("    ADD(a,b) = %i\n", swigexample.do_op(a,b,swigexample.ADD));
printf("    SUB(a,b) = %i\n", swigexample.do_op(a,b,swigexample.SUB));
printf("    MUL(a,b) = %i\n", swigexample.do_op(a,b,swigexample.MUL));

printf("Here is what the C callback function objects look like in Octave\n");
swigexample.ADD
swigexample.SUB
swigexample.MUL

printf("Call the functions directly...\n");
printf("    add(a,b) = %i\n", swigexample.add(a,b));
printf("    sub(a,b) = %i\n", swigexample.sub(a,b));
