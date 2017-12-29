lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

// Try to set the values of some global variables
ivar_set(42);
svar_set(-31000);
lvar_set(65537);
uivar_set(uint32(123456));
usvar_set(uint16(61000));
ulvar_set(654321);
scvar_set(int8(-13));
ucvar_set(uint8(251));
cvar_set("S");
fvar_set(3.14159);
dvar_set(2.1828);
strvar_set("Hello World");
iptrvar_set(new_int(37));
ptptr_set(new_Point(37,42));
name_set("Bill");

// Now print out the values of the variables
printf("Variables (values printed from Scilab)\n");
printf("ivar      = %i\n", ivar_get());
printf("svar      = %i\n", svar_get());
printf("lvar      = %i\n", lvar_get());
printf("uivar     = %i\n", uivar_get());
printf("usvar     = %i\n", usvar_get());
printf("ulvar     = %i\n", ulvar_get());
printf("scvar     = %i\n", scvar_get());
printf("ucvar     = %i\n", ucvar_get());
printf("fvar      = %f\n", fvar_get());
printf("dvar      = %f\n", dvar_get());
printf("cvar      = %s\n", cvar_get());
printf("strvar    = %s\n", strvar_get());
printf("cstrvar   = %s\n", cstrvar_get());
printf("iptrvar   = %i\n", value_int(iptrvar_get()));
printf("name      = %s\n", name_get());
printf("ptptr     = %s\n", Point_print(ptptr_get()));
printf("pt        = %s\n", Point_print(pt_get()));
printf("status    = %d\n", status_get());

printf("\nVariables (values printed from C)\n");
print_vars()

// Immutable variables
printf("\nNow I''m going to try and modify some read only variables\n");
printf("     Tring to set ''path''\n");
try
    path_set("Whoa!");
    printf("Hey, what''s going on?!?! This shouldn''t work\n");
catch
    printf("Good.\n");
end
printf("     Trying to set ''status''\n");
try
    status_set(0);
    printf("Hey, what''s going on?!?! This shouldn''t work\n");
catch
    printf("Good.\n");
end

// Structure
printf("\nI''m going to try and update a structure variable.\n");
pt_set(ptptr_get());
printf("The new value is %s\n", Point_print(pt_get()));

exit

