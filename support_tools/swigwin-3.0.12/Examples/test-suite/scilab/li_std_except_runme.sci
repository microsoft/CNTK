exec('swigtest.start', -1);

function checkException(cmd, expected_ierr, expected_error_msg)
  ierr = execstr(cmd, 'errcatch');
  checkequal(ierr, expected_ierr, cmd + ': ierr');
  checkequal(lasterror(), 'SWIG/Scilab: ' + expected_error_msg, cmd + ': msg');
endfunction

t = new_Test();

checkException('Test_throw_bad_exception(t)', 20010, 'SystemError: std::bad_exception');

checkException('Test_throw_domain_error(t)', 20009, 'ValueError: oops');

checkException('Test_throw_exception(t)', 20010, 'SystemError: std::exception');

checkException('Test_throw_invalid_argum(t)', 20009, 'ValueError: oops');

checkException('Test_throw_length_error(t)', 20004, 'IndexError: oops');

checkException('Test_throw_logic_error(t)', 20003, 'RuntimeError: oops');

checkException('Test_throw_out_of_range(t)', 20004, 'IndexError: oops');

checkException('Test_throw_overflow_erro(t)', 20007, 'OverflowError: oops');

checkException('Test_throw_range_error(t)', 20007, 'OverflowError: oops');

checkException('Test_throw_runtime_error(t)', 20003, 'RuntimeError: oops');

delete_Test(t);

exec('swigtest.quit', -1);
