%module contract

%warnfilter(SWIGWARN_RUBY_MULTIPLE_INHERITANCE,
	    SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) C; /* Ruby, C#, D, Java, PHP multiple inheritance */

#ifdef SWIGCSHARP
%ignore B::bar; // otherwise get a warning: `C.bar' no suitable methods found to override
#endif

#ifdef SWIGD
%ignore B::bar; // Prevents getting an error that C.bar does not override any function because multiple inheritance is not supported.
#endif

%contract test_preassert(int a, int b) {
require:
	a > 0;
	b > 0;
}

%contract test_postassert(int a) {
ensure:
	test_postassert > 0;
}

%contract test_prepost(int a, int b) {
require:
	a > 0;
ensure:
	test_prepost > 0;
}

%inline %{

int test_preassert(int x, int y) {
   if ((x > 0) && (y > 0)) return 1;
   return 0;
}

int test_postassert(int x) {
    return x;
}

int test_prepost(int x, int y) {
    return x+y;    
}
%}

/* Class tests */

%contract Foo::test_preassert(int x, int y) {
 require:
  x > 0;
  y > 0;
}

%contract Foo::test_postassert(int a) {
 ensure:
  test_postassert > 0;
}

%contract Foo::test_prepost(int a, int b) {
 require:
  a > 0;
 ensure:
  test_prepost > 0;
}

%contract Foo::stest_prepost(int a, int b) {
 require:
  a > 0;
 ensure:
  stest_prepost > 0;
}

%contract Bar::test_prepost(int c, int d) {
 require:
  d > 0;
}

%inline %{
class Foo {
public:
        virtual ~Foo() { }
  
	virtual int test_preassert(int x, int y) {
            if ((x > 0) && (y > 0)) return 1;
            return 0;
	}   
	virtual int test_postassert(int x) {
	  return x;
	}
	virtual int test_prepost(int x, int y) {
	  return x+y;
	}
	static int stest_prepost(int x, int y) {
	  return x+y;
	}
 };

class Bar : public Foo {
public:
	virtual int test_prepost(int x, int y) {
	  return x+y;
	}
};

%}

/* Multiple inheritance test */

%contract A::foo(int i, int j, int k, int l, int m) {
 require:
  i > 0;
  j > 0;
 ensure:
  foo > 0;
}

%contract B::bar(int x, int y, int z, int w, int v) {
 require:
  w > 0;
  v > 0;
 ensure:
  bar > 0;
}

%contract C::foo(int a, int b, int c, int d, int e) {
 require:
  c > 0;
  d > 0;
 ensure:
  foo > 0;
}

%contract D::foo(int, int, int, int, int x) {
 require:
  x > 0;
}

%contract D::bar(int a, int b, int c, int, int) {
 require:
  a > 0;
  b > 0;
  c > 0;
}

%inline %{
  class A {
   public:
    virtual ~A() {}
    virtual int foo(int a, int b, int c, int d, int e) {
      if ((a > 0) && (b > 0) && (c > 0) && (d > 0) && (e > 0)) {
	return 1;
      }
      return 0;
    }
  };

  class B {
   public:
    virtual ~B() {}
    virtual int bar(int a, int b, int c, int d, int e) {
      if ((a > 0) && (b > 0) && (c > 0) && (d > 0) && (e > 0)) {
	return 1;
      }
      return 0;
    }
  };

  class C : public A, public B {
   public:
    virtual int foo(int a, int b, int c, int d, int e) {
      return A::foo(a,b,c,d,e);
    }
    virtual int bar(int a, int b, int c, int d, int e) {
      return B::bar(a,b,c,d,e);
    }
  };
  
  class D : public C {
   public:
    virtual int foo(int a, int b, int c, int d, int e) {
      return C::foo(a,b,c,d,e);
    }
    virtual int bar(int a, int b, int c, int d, int e) {
      return C::bar(a,b,c,d,e);
    }
  };
  %}

%extend E {
  %contract manipulate_i(int i) {
  require:
  i <= $self->m_i;
  }
}

%inline %{
struct E {
  int m_i;
  void manipulate_i(int i) {
  }
};
%}


// Namespace

%{
namespace myNames {

class myClass
{
    public:
    	myClass(int i) {}
};

}
%}

namespace myNames {

%contract myClass::myClass( int i ) {
require:
    i > 0;
}

class myClass
{
    public:
    	myClass(int i) {}
};

}

