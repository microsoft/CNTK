# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

swigexample

# ----- Object creation -----

# Print out the value of some enums
printf("*** color ***\n");
printf("    RED    = %i\n", swigexample.RED);
printf("    BLUE   = %i\n", swigexample.BLUE);
printf("    GREEN  = %i\n", swigexample.GREEN);

printf("\n*** Foo::speed ***\n");
printf("    Foo_IMPULSE   = %i\n", swigexample.Foo_IMPULSE);
printf("    Foo_WARP      = %i\n", swigexample.Foo_WARP);
printf("    Foo_LUDICROUS = %i\n", swigexample.Foo_LUDICROUS);

printf("\nTesting use of enums with functions\n");

swigexample.enum_test(swigexample.RED, swigexample.Foo_IMPULSE);
swigexample.enum_test(swigexample.BLUE,  swigexample.Foo_WARP);
swigexample.enum_test(swigexample.GREEN, swigexample.Foo_LUDICROUS);
swigexample.enum_test(1234,5678)

printf("\nTesting use of enum with class method\n");
f = swigexample.Foo();

f.enum_test(swigexample.Foo_IMPULSE);
f.enum_test(swigexample.Foo_WARP);
f.enum_test(swigexample.Foo_LUDICROUS);
