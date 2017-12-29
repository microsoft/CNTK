%module refcount

%warnfilter(SWIGWARN_IGNORE_OPERATOR_EQ,SWIGWARN_LANG_IDENTIFIER);

%{ 
#include <iostream> 
#include "refcount.h"
%}

//
// using the %refobject/%unrefobject directives you can activate the
// reference counting for RCObj and all its descendents at once
//

%refobject   RCObj "$this->addref();"
%unrefobject RCObj "$this->delref();"

%include "refcount.h"

%newobject B::create(A* a);
%newobject global_create(A* a);
%newobject B::cloner();
%newobject Factory::create(A* a);
%newobject Factory::create2(A* a);


 
%inline %{

  struct A : RCObj
  {
    A() {}
    
    ~A() 
    {
      // std::cout << "deleting a" << std::endl;
    }

#ifdef SWIGRUBY 
    // fix strange ruby + virtual derivation problem
    using RCObjBase::ref_count;
#endif
  };

  struct A1 : A 
  {
  protected:
    A1() {}
  };

  struct A2 : A
  {
  };

  struct A3 : A1, private A2
  {    
  };

%}

#if defined(SWIGPYTHON)
%extend_smart_pointer(RCPtr<A>);
%template(RCPtr_A) RCPtr<A>;
#endif

%inline %{
  
  struct B : RCObj
  {
    B(A* a) : _a(a) {}
    
    A* get_a() 
    {
      return _a;
    }
    
    static B* create(A* a)
    {
      return new B(a);
    }
    
    B* cloner() 
    {
      return new B(_a);
    }

    ~B() 
    {
      // std::cout << "deleting b" << std::endl;
    }

    RCPtr<A> get_rca() {
      return _a;      
    }

  private:
    RCPtr<A> _a;
  };

struct B* global_create(A* a)
{
  return new B(a);
}

struct Factory {
  static B* create(A* a)
  {
    return new B(a);
  }
  B* create2(A* a)
  {
    return new B(a);
  }
};

%}

#if defined(SWIGPYTHON) || defined(SWIGOCTAVE)

%include <std_vector.i>
%template(vector_A) std::vector<RCPtr<A> >;

#endif
