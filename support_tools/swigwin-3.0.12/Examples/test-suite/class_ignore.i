%module class_ignore

%ignore Foo;
%ignore *::Bar::foo;
%ignore Far::away() const;

%inline %{
  class Foo {
  public:
    virtual ~Foo() { }
    virtual char *blah() = 0;
  };
  
  namespace hi 
  {
    namespace hello
    {
      class Bar : public Foo {
      public:
	void foo(void) {};
	
	virtual char *blah() { return (char *) "Bar::blah"; }
      };

    }
  }

  struct Boo {
    virtual ~Boo() {}
    virtual void away() const {}
  };
  struct Far : Boo {
    virtual void away() const {}
  };
  struct Hoo : Far {
    virtual void away() const {}
  };

  char *do_blah(Foo *f) {
    return f->blah();
  }

  class ForwardClass;  
  template <class C> class ForwardClassT;  
  template<typename T1, typename T2> class PatchList;
%}

