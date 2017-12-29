lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end
example_Init();

// This example shows how to use C++ fonctions with STL lists arguments
// Here, STL lists are converted from/to Scilab matrices (SWIG_SCILAB_EXTRA_NATIVE_CONTAINERS is not defined)

// integer lists

disp("Example of passing matrices of int as list arguments of C++ functions.");
disp("get a list of int {1...4} from create_integer_list():");
is = create_integer_list(1, 4);
disp(is);
disp("get the sum of this list elements with sum_integer_list():")
sum = sum_integer_list(is);
disp(is);
is2 = create_integer_list(3, 6);
disp("concat this list with the list of int {3...6} with concat_integer_list():");
is3 = concat_integer_list(is, is2);
disp(is3);

// string lists

disp("Example of passing matrices of string as list arguments of C++ functions.");
disp("get a list of string {''aa'', ''bb'', ''cc'', ''dd''} with create_string_list():");
ss = create_string_list("aa bb cc dd");
disp(ss);
ss2 = create_string_list("cc dd ee ff");
disp("concat this list with the list of string {''cc'', ''dd'', ''ee'', ''ff''} with concat_string_list():");
ss3 = concat_string_list(ss, ss2);
disp(ss3);

exit

