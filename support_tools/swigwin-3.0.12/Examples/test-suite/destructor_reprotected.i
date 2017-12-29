%module destructor_reprotected


%inline {

  struct A
  {
    A()
    {
    }
    
    virtual ~A()
    {
    }
    
  };
  
  struct B : A
  {
  protected:
    B()
    {
    }
    
    ~B()
    {
    }
    
  };

  struct C : B
  {
    C()
    {
    }
    
    ~C()
    {
    }    
  };
}
