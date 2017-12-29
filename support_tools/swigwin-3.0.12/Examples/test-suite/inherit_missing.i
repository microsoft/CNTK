// Tests handling of inheritance when a base class isn't provided to SWIG
%module inherit_missing

%warnfilter(402);

%{
/* Define the class internally, but don't tell SWIG about it */
class Foo {
public:
     virtual ~Foo() {}
     virtual char *blah() {
	return (char *) "Foo::blah";
     }
};
%}

/* Forward declaration.  Says that Foo is a class, but doesn't provide a definition */

class Foo;

%newobject new_Foo;
%delobject delete_Foo;
%inline %{

class Bar : public Foo {
 public:
  virtual ~Bar() {}
  virtual char *blah() {
    return (char *) "Bar::blah";
  };
};

class Spam : public Bar {
  public:
  virtual char *blah() {
    return (char *) "Spam::blah";
  };
};

Foo *new_Foo() {
   return new Foo();
}

void delete_Foo(Foo *f) {
   delete f;
}

char *do_blah(Foo *f) {
  return f->blah();
}
%}
