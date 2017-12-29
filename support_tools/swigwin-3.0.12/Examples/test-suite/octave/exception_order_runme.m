# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

exception_order

function check_lasterror(expected)
  if (!strcmp(lasterror.message, expected))
    # Take account of older versions prefixing with "error: " and adding a newline at the end
    if (!strcmp(regexprep(lasterror.message, 'error: (.*)\n$', '$1'), expected))
      error(["Bad exception order. Expected: \"", expected, "\" Got: \"", lasterror.message, "\""])
    endif
  endif
endfunction

a = A();

try
  a.foo()
catch
  check_lasterror("C++ side threw an exception of type E1")
end_try_catch

try
  a.bar()
catch
  check_lasterror("C++ side threw an exception of type E2")
end_try_catch

try
  a.foobar()
catch
  check_lasterror("postcatch unknown (SWIG_RuntimeError)")
end_try_catch

try
  a.barfoo(1)
catch
  check_lasterror("C++ side threw an exception of type E1")
end_try_catch

try
  a.barfoo(2)
catch
  check_lasterror("C++ side threw an exception of type E2 *")
end_try_catch
