lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end
example_Init();

printf("\nTest constants\n");
printf("ICONST  = %i (should be 42)\n", ICONST);
printf("FCONST  = %5.4f (should be 2.1828)\n", FCONST);
printf("SCONST  = ''%s'' (should be ''Hello World'')\n", SCONST);
printf("EXPR    = %5.4f (should be 48.5484)\n", EXPR);
printf("iconst  = %i (should be 37)\n", iconst);
printf("fconst  = %3.2f (should be 3.14)\n", fconst);

exit
