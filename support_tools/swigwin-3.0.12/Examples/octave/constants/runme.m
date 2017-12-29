# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

swigexample

printf("ICONST  = %i (should be 42)\n", swigexample.ICONST);
printf("FCONST  = %f (should be 2.1828)\n", swigexample.FCONST);
printf("CCONST  = %s (should be 'x')\n", swigexample.CCONST);
printf("CCONST2 = %s (this should be on a new line)\n", swigexample.CCONST2);
printf("SCONST  = %s (should be 'Hello World')\n", swigexample.SCONST);
printf("SCONST2 = %s (should be '\"Hello World\"')\n", swigexample.SCONST2);
printf("EXPR    = %f (should be 48.5484)\n", swigexample.EXPR);
printf("iconst  = %i (should be 37)\n", swigexample.iconst);
printf("fconst  = %f (should be 3.14)\n", swigexample.fconst);

try
    printf("EXTERN = %s (Arg! This shouldn't printf(anything)\n", swigexample.EXTERN);
catch
    printf("EXTERN isn't defined (good)\n");
end_try_catch

try
    printf("FOO    = %i (Arg! This shouldn't printf(anything)\n", swigexample.FOO);
catch
    printf("FOO isn't defined (good)\n");
end_try_catch
