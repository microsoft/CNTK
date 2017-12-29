/* File : example.i */
%module template_inherit

/* This example tests template inheritance to see if it actually works */

%inline %{

template<class T> class Foo {
public:
  virtual ~Foo() { }
  virtual char *blah() {
       return (char *) "Foo";
  }
  virtual char *foomethod() {
       return (char *) "foomethod";
  }
};

template<class T> class Bar : public Foo<T> {
public:
   virtual char *blah() {
        return (char *) "Bar";
   }
};

template<class T> char *invoke_blah(Foo<T> *x) {
   return x->blah();
}
%}

%template(FooInt) Foo<int>;
%template(FooDouble) Foo<double>;
%template(FooUInt) Foo<unsigned int>;
%template(BarInt) Bar<int>;
%template(BarDouble) Bar<double>;
%template(BarUInt) Bar<unsigned>;
%template(invoke_blah_int) invoke_blah<int>;
%template(invoke_blah_double) invoke_blah<double>;
%template(invoke_blah_uint) invoke_blah<int unsigned>;


