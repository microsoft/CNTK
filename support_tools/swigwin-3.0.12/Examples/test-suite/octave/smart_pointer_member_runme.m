smart_pointer_member

f = Foo();
f.y = 1;

if (f.y != 1)
  error
endif

b = Bar(f);
b.y = 2;

if (f.y != 2)
  error("f.y = %i, b.y = %i",f.y,b.y)
endif

if (swig_this(b.x) != swig_this(f.x))
  error
endif

if (b.z != f.z)
  error
endif

try
  if (Foo.z == Bar.z)
    error
  endif
    error
catch
end_try_catch

  






