nondynamic

aa = nondynamic.A();

aa.a = 1;
aa.b = 2;
try
  aa.c = 2;
  err = 0;
catch
  err = 1;
end_try_catch

if (!err)
  error("A is not static")
endif


B=@() subclass(nondynamic.A(),'c',4);

bb = B();
bb.c = 3;
try
  bb.d = 2
  err = 0
catch
  err = 1
end_try_catch

if (!err)
  error("B is not static")
endif
    
cc = nondynamic.C();
cc.d = 3;

