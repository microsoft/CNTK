lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

// First create some objects using the pointer library.
printf("Testing the pointer library\n")
a = new_intp();
b = new_intp();
c = new_intp(); // Memory for result

intp_assign(a, 37);
intp_assign(b, 42);

printf("     a = %d\n", intp_value(a));
printf("     b = %d\n", intp_value(b));
printf("     c = %d\n", intp_value(c));

// Call the add() function with some pointers
add(a, b, c);

// Now get the result
r = intp_value(c);
printf("     37 + 42 = %d\n", r);

// Clean up the pointers
delete_intp(a);
delete_intp(b);
delete_intp(c);

// Now try the typemap library
// This should be much easier. Now how it is no longer
// necessary to manufacture pointers.
printf("Trying the typemap library\n");
r = sub(37, 42);
printf("     37 - 42 = %d\n", r);

// Now try the version with multiple return values
printf("Testing multiple return values\n");
[q, r] = divide(42, 37);
printf("     42/37 = %d remainder %d\n", q, r);

exit

