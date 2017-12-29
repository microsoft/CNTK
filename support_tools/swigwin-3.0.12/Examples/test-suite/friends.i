%module friends
%{
#include <iostream>
%}

%warnfilter(SWIGWARN_LANG_IDENTIFIER);

#if defined(SWIGOCTAVE)
%warnfilter(SWIGWARN_IGNORE_OPERATOR_LSHIFT_MSG) operator<<;
%warnfilter(SWIGWARN_IGNORE_OPERATOR_RSHIFT_MSG) operator>>;
#endif


%inline 
{

  void globalscope(); // forward declaration needed for some compilers

  struct A;
  struct B
  {
    B(int i) : v(i) 
    {
    }
    
    friend void ::globalscope();
    friend int mix(A* a, B *b);
    virtual ~B()
    {
    }
    
  private:
    int v;
    
  };
  
  void globalscope() { B b(0); b.v=10; }
  
  struct A
  {
    A(int v) : val(v)
    {
    }

    friend int get_val1(const A& a)
    {
      return a.val;
    }

    /* simple overloading */
    friend int get_val1(const A& a, int o)
    {
      return a.val + o;
    }

    /*
      note that operators << and >> are ignored, as they
      should, since no rename is performed.
    */
    friend std::istream& operator>>(std::istream& in, A& a);

    /* already declare at B */
    friend int mix(A* a, B *b);

  protected:
    friend int get_val2(const A& a)
    {
      return a.val*2;
    }
    
  private:
    friend int get_val3(const A& a);

    /* this should be ignored */
    friend std::ostream& operator<<(std::ostream& out, const A& a)
    {
      out << a.val;
      return out;
    }

    int val;    
  };

  /* 
     'mix' is an interesting case, this is the third declaration
     swig is getting (two friends + one inline).
   */
  inline int mix(A* a, B *b) {
    return a->val + b->v;
  }

  /* this should be ignored */
  inline std::istream& operator>>(std::istream& in, A& a) {
    int v;
    in >> v;
    a = A(v);
    return in;
  }

  inline int get_val3(const A& a) {
    return a.val*3;
  }
  
  /* another overloading */
  inline int get_val1(int i, int a, int b) {
    return i;
  }


  /*
    sit and watch how well this case works, is just incredible!!,

    also note that there is no special code added to manage friends
    and templates (or overloading), this is just old swig magic
    working at its best.
  */

  template <class C>
    struct D
    {
      D(C v) : val(v) {}

      /* note that here we are overloading the already super
	 overloaded 'get_val1' */
      friend C get_val1(D& b)
      {
	return b.val;
      }

      /* here set will be 'auto' overloaded, depending of the
	 %template instantiations. */
      friend void set(D& b, C v)
      {
	b.val = v;
      }

    private:
      C val;
    };

  namespace ns1 {

    void bas() {}

    void baz() {}
  }
}

// Use this version with extra qualifiers to test SWIG as some compilers accept this
  namespace ns1 {
    namespace ns2 {
      class Foo {
      public:
	Foo::Foo() {};
	friend void bar();
	friend void ns1::baz();	
	void Foo::member() { }
	
      };
      void bar() {}    
    }
  }

// Remove extra qualifiers for the compiler as some compilers won't compile the extra qaulification (eg gcc-4.1 onwards) 
%{
  namespace ns1 {
    namespace ns2 {
      class Foo {
      public:
	Foo() {};
	friend void bar();
	friend void ns1::baz();	
	void member() { }
	
      };
      void bar() {}    
    }
  }
%}
    

%template(D_i) D<int>;
%template(D_d) D<double>;
