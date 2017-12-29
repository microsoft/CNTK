lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

// Test use of a struct (Bar)

a = new_Bar();

Bar_x_set(a, 100);
printf("a.x = %d (Should be 100)\n", Bar_x_get(a));

delete_Bar(a);

exit
