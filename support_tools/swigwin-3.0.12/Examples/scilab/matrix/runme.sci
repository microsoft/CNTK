lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

// create a new matrix
x = new_matrix();
for i = 0 : 3;
  for j = 0 : 3;
    set_m(x, i, j, i+j);
  end;
end;

// print the matrix
print_matrix(x);

// another matrix
y = new_matrix();
  for i = 0 : 3;
    for j = 0 : 3;
      set_m(y, i, j, i-j);
    end;
  end;

// print the matrix
print_matrix(y);

// mat_mult the two matrix, and the result is stored in a new matrix
z = new_matrix();

mat_mult(x, y, z);

print_matrix(z);

//destroy the matrix
destroy_matrix(x);
destroy_matrix(y);
destroy_matrix(z);

exit
