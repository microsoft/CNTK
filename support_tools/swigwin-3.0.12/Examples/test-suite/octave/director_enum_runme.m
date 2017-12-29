director_enum

MyFoo=@() subclass(director_enum.Foo(),'say_hi',@(self,val) val);

b = director_enum.Foo();
a = MyFoo();

if (a.say_hi(director_enum.hello) != b.say_hello(director_enum.hi))
  error
endif
