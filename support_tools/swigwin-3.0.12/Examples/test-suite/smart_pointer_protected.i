%module smart_pointer_protected

%inline %{

  namespace hi
  {    
    struct A 
    {
      virtual ~A() { }
      virtual int value(A*) = 0;
      int index;
    };    
    
    struct B : A 
    {
    protected:
      int value(A*)
      {
	return 1;
      }
    };

    struct C
    {
      hi::B* operator->() const { return new hi::B(); }
    private:
      int index;
    };
  }
  

%}

