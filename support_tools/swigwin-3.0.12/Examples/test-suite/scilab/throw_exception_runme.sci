exec("swigtest.start", -1);

function checkException(cmd, expected_error_msg)
  ierr = execstr(cmd, 'errcatch');
  checkequal(ierr, 20000, cmd + ': ierr');
  checkequal(lasterror(), 'SWIG/Scilab: ' + expected_error_msg, cmd + ': msg');
endfunction

foo = new_Foo();

checkException('Foo_test_int(foo)', 'Exception (int) occured: 37');

checkException('Foo_test_msg(foo)', 'Exception (char const *) occured: Dead');

checkException('Foo_test_multi(foo, 1)', 'Exception (int) occured: 37');

checkException('Foo_test_multi(foo, 2)', 'Exception (char const *) occured: Dead');

checkException('Foo_test_cls(foo)', 'Exception (CError) occured.');

delete_Foo(foo);

exec("swigtest.quit", -1);
