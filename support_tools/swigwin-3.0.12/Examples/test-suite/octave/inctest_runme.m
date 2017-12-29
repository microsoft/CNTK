inctest

try
  a = inctest.A();
catch
  error("didn't find A\ntherefore, I didn't include \
      'testdir/subdir1/hello.i'")
end_try_catch

try
  b = inctest.B();
catch
  error("didn't find B\ntherefore, I didn't include 'testdir/subdir2/hello.i'")
end_try_catch

# Check the import in subdirectory worked
if (inctest.importtest1(5) != 15)
  error("import test 1 failed")
endif

if (!strcmp(inctest.importtest2("black"),"white"))
  error("import test 2 failed")
endif

