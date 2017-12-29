director_nested

A=@() subclass(FooBar_int(),
	       'do_step',@(self) "A::do_step;",
	       'get_value',@(self) "A::get_value");

a = A();
if (!strcmp(a.step(),"Bar::step;Foo::advance;Bar::do_advance;A::do_step;"))
  error("Bad A virtual resolution")
endif

B=@() subclass(FooBar_int(),
	       'do_advance',@(self) strcat("B::do_advance;",self.do_step()),
	       'do_step',@(self) "B::do_step;",
	       'get_value',@(self) 1);

b = B();

if (!strcmp(b.step(),"Bar::step;Foo::advance;B::do_advance;B::do_step;"))
  error("Bad B virtual resolution")
endif

C=@() subclass(FooBar_int(),
	       'do_advance',@(self) strcat("C::do_advance;",self.FooBar_int.do_advance()),
	       'do_step',@(self) "C::do_step;",
	       'get_value',@(self) 2,
	       'get_name',@(self) strcat(self.FooBar_int.get_name()," hello"));

cc = C();
c = FooBar_int_get_self(cc);
c.advance();

if (!strcmp(c.get_name(),"FooBar::get_name hello"))
  error
endif

if (!strcmp(c.name(),"FooBar::get_name hello"))
  error
endif
