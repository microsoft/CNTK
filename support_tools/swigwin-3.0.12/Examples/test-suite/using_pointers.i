%module using_pointers

#ifdef SWIGCSHARP
%csmethodmodifiers x "public new"
#endif

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%inline %{
  class Foo {
  public:
    int x;
    virtual ~Foo() { }
    virtual Foo* blah() { return this; }
    virtual Foo* exception_spec(int what_to_throw) throw (int, const char *) {
      int num = 10;
      const char *str = "exception message";
      if (what_to_throw == 1) throw num;
      else if (what_to_throw == 2) throw str;
      return 0;
    }
  };

  class FooBar : public Foo {
  public:
    using Foo::blah;
    using Foo::x;
    using Foo::exception_spec;
  };

%}

