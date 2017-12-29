%module disown

%{
#include <iostream>
%}

#pragma SWIG nowarn=SWIGWARN_TYPEMAP_APPLY_UNDEF

%apply SWIGTYPE *DISOWN { A *disown };

%inline {
  struct A
  {
    ~A()
    {
      // std::cout <<"delete A" << std::endl;
    }
    
    
  };
  
  class B
  {
    A *_a;
  public:
    B() : _a(0)
    {
    }
    
    ~B()
    {
      if (_a) {
	// std::cout <<"delete A from B" << std::endl;	
	delete _a;
      }
      // std::cout <<"delete B" << std::endl;      
    }

    int acquire(A *disown) 
    {
      // std::cout <<"acquire A" << std::endl;      
      _a = disown;    
      return 5;      
    }

    int remove(A *remove) 
    {
      delete remove;
      return 5;      
    }
    
  };
}
