# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

swigexample

# Try to set the values of some global variables

swigexample.cvar.ivar   =  42;
swigexample.cvar.svar   = -31000;
swigexample.cvar.lvar   =  65537;
swigexample.cvar.uivar  =  123456;
swigexample.cvar.usvar  =  61000;
swigexample.cvar.ulvar  =  654321;
swigexample.cvar.scvar  =  -13;
swigexample.cvar.ucvar  =  251;
swigexample.cvar.cvar   =  "S";
swigexample.cvar.fvar   =  3.14159;
swigexample.cvar.dvar   =  2.1828;
swigexample.cvar.strvar =  "Hello World";
swigexample.cvar.iptrvar= swigexample.new_int(37);
swigexample.cvar.ptptr  = swigexample.new_Point(37,42);
swigexample.cvar.name   = "Bill";

# Now print out the values of the variables

printf("Variables (values printed from Octave)\n");

printf("ivar      = %i\n", swigexample.cvar.ivar);
printf("svar      = %i\n", swigexample.cvar.svar);
printf("lvar      = %i\n", swigexample.cvar.lvar);
printf("uivar     = %i\n", swigexample.cvar.uivar);
printf("usvar     = %i\n", swigexample.cvar.usvar);
printf("ulvar     = %i\n", swigexample.cvar.ulvar);
printf("scvar     = %i\n", swigexample.cvar.scvar);
printf("ucvar     = %i\n", swigexample.cvar.ucvar);
printf("fvar      = %i\n", swigexample.cvar.fvar);
printf("dvar      = %i\n", swigexample.cvar.dvar);
printf("cvar      = %s\n", swigexample.cvar.cvar);
printf("strvar    = %s\n", swigexample.cvar.strvar);
#printf("cstrvar   = %s\n", swigexample.cvar.cstrvar);
swigexample.cvar.iptrvar
printf("name      = %i\n", swigexample.cvar.name);
printf("ptptr     = %s\n", swigexample.Point_print(swigexample.cvar.ptptr));
#printf("pt        = %s\n", swigexample.cvar.Point_print(swigexample.cvar.pt));

printf("\nVariables (values printed from C)\n");

swigexample.print_vars();

printf("\nNow I'm going to try and modify some read only variables\n");

printf("     Tring to set 'path'\n");
try
    swigexample.cvar.path = "Whoa!";
    printf("Hey, what's going on?!?! This shouldn't work\n");
catch
    printf("Good.\n");
end_try_catch

printf("     Trying to set 'status'\n");
try
    swigexample.cvar.status = 0;
    printf("Hey, what's going on?!?! This shouldn't work\n");
catch
    printf("Good.\n");
end_try_catch


printf("\nI'm going to try and update a structure variable.\n");

swigexample.cvar.pt = swigexample.cvar.ptptr;

printf("The new value is %s\n", swigexample.Point_print(swigexample.cvar.pt));
