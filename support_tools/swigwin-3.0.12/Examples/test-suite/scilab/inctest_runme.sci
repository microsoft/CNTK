exec("swigtest.start", -1);

try
  a = new_A();
catch
  printf("did not find A\ntherefore, I did not include ""testdir/subdir1/hello.i""\n");
  swigtesterror();
end

try
  b = new_B();
catch
  printf("did not find B\ntherefore, I did not include ""testdir/subdir2/hello.i""\n");
  swigtesterror();
end

if importtest1(5) <> 15 then swigtesterror(); end
if importtest2("black") <> "white" then swigtesterror(); end

exec("swigtest.quit", -1);
