lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end
example_Init();

// Call some templated functions
printf("maxint(3, 7) = %i\n", maxint(3, 7));
printf("maxdouble(3.14, 2.18) = %3.2f\n", maxdouble(3.14, 2.18));

// Create some class

iv = new_vecint(100);
dv = new_vecdouble(1000);

for i = 0:100
  vecint_setitem(iv, i, 2*i);
end

for i = 0:100
  vecdouble_setitem(dv, i, 1.0/(i+1));
end

isum = 0
for i = 0:100
    isum = isum + vecint_getitem(iv, i);
end

printf("isum = %i\n", isum);

dsum = 0
for i = 0:100
    dsum = dsum + vecdouble_getitem(dv, i);
end

printf("dsum = %3.2f\n", dsum);

delete_vecint(iv);
delete_vecdouble(dv);

exit

