return_const_value

p = return_const_value.Foo_ptr.getPtr();
if (p.getVal() != 17)
  error("Runtime test1 faild. p.getVal()=", p.getVal())
endif

p = return_const_value.Foo_ptr.getConstPtr();
if (p.getVal() != 17)
  error("Runtime test2 faild. p.getVal()=", p.getVal())
endif
