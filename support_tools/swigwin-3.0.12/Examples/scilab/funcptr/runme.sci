lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

a = 37
b = 42

// Now call our C function with a bunch of callbacks

printf("Trying some C callback functions\n");
printf("    a        = %i\n", a);
printf("    b        = %i\n", b);
printf("    ADD(a,b) = %i\n", do_op(a,b,funcvar_get()));

exit



