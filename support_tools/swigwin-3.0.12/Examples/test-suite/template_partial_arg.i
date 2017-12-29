%module template_partial_arg

%inline %{
  template <class T> class Foo {
  public: 
    T bar() { return T(); }  
    T* baz() { return 0; } 
  };

  template <class T> class Foo<T*> {
  public: 
    T bar() { return T(); }
    T* baz() { return 0; }
  };

  class Bar {};
%}

%template(Foo1) Foo<Bar>;
%template(Foo2) Foo<Bar*>;

