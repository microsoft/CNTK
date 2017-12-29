// C++ namespace tests

%module cpp_namespace

%inline %{
  typedef int Bad;

  /* A very basic namespace */
  namespace example {
    typedef char *Bad;

    int fact(int n) {
      if (n <= 0) return 1;
      else return n*fact(n-1);
    }
    int Foo = 42;

    class Test {
    public:
      Test() { }
      ~Test() { }
      char *method() {
	return (char *) "Test::method";
      }
    };
    typedef Test *TestPtr;
    void weird(Bad x, ::Bad y) { }
  }

  char *do_method(example::TestPtr t) {
    return t->method();
  }

  namespace ex = example;

  char *do_method2(ex::TestPtr t) {
     return t->method();
  }

%}

// Some more complicated namespace examples

%inline %{
namespace Foo {
   typedef int Integer;
   class Test2 { 
   public:
       virtual ~Test2() { }
       virtual char *method() {	
	 return (char *) "Test2::method";
       }
   };
  typedef Test2 *Test2Ptr;
}

namespace Foo2 {
  using Foo::Integer;
  using Foo::Test2;
  class Test3 : public Test2 {
  public:
    virtual char *method() {	
      return (char *) "Test3::method";
    }
  };
  typedef Test3 *Test3Ptr;
  typedef Test3 Test3Alt;
}

namespace Foo3 {
  using namespace Foo2;
  class Test4 : public Test3 {
  public:
    virtual char *method() {	
      return (char *) "Test4::method";
    }
  };
  Integer foo3(Integer x) { return x; }
  typedef Test4 *Test4Ptr;
  
}
   
using Foo2::Test3Alt;
using Foo3::Integer;

class Test5 : public Test3Alt { 
public:
  virtual char *method() {	
    return (char *) "Test5::method";
  }
};

char *do_method3(Foo::Test2 *t, Integer x) {
  return t->method();
}

%}



   


      	

