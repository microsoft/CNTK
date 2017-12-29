%module typemap_self

// This typemap should be ignored for self?
%typemap(in) A* (A* ptr) {
  if (SWIG_ConvertPtr($input, (void**) &ptr, $1_descriptor, 0) != -1) {
    $1 = ptr;
  } else  {
    $1 = new A();
  }
 }

// Simple but unsecure current fix
//%apply SWIGTYPE* {A* self}


%inline %{
  class A;
  
  int foo(A* self) 
  {
    return 0;
  }
  
  struct A
  {
    static int bar(int, A* self)
    {
      return 1;
    }
    
    int val;
    
    
    int foo(A* self, A* b) 
    {
      return 1;
    }
  };
  
  struct B
  {
    B(A*) 
    {
    }
  };
  
%}
