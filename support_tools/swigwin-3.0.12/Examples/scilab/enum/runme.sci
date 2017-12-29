lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end
example_Init();

printf("\nTest enums\n");
printf("*** color ***\n");
printf("    RED_get()    = %i\n", RED_get());
printf("    BLUE_get()   = %i\n", BLUE_get());
printf("    GREEN_get()  = %i\n", GREEN_get());

printf("\n*** Foo::speed ***\n")
printf("    Foo_IMPULSE   = %i\n", Foo_IMPULSE_get());
printf("    Foo_WARP      = %i\n", Foo_WARP_get());
printf("    Foo_LUDICROUS = %i\n", Foo_LUDICROUS_get());

printf("\nTest enums as argument of functions\n");

enum_test(RED_get(), Foo_IMPULSE_get());
enum_test(BLUE_get(), Foo_WARP_get());
enum_test(GREEN_get(), Foo_LUDICROUS_get());
enum_test(1234, 5678);

printf("\nTest enums as argument of class methods\n");

f = new_Foo();
Foo_enum_test(f, Foo_IMPULSE_get());
Foo_enum_test(f, Foo_WARP_get());
Foo_enum_test(f, Foo_LUDICROUS_get());
delete_Foo(f);

exit
