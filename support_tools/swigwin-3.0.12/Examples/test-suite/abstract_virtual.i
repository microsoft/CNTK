%module(ruby_minherit="1") abstract_virtual

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) D; /* C#, D, Java, PHP multiple inheritance */
%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) E; /* C#, D, Java, PHP multiple inheritance */

%inline %{
#if defined(_MSC_VER)
  #pragma warning( disable : 4250) // warning C4250: 'D' : inherits 'B::B::foo' via dominance
#endif
  struct A 
  {
    virtual ~A()
    {
    }
    
    virtual int foo() = 0;
  };
 
  struct B : virtual A
  {
    int foo() 
    {
      return 0;
    }
  };
  
  struct C: virtual A
  {
  protected:
    C()
    {
    }
  };

  //
  // This case works
  //
  struct D : B, C
  {
    D()
    {
    }
  };

  //
  // This case doesn't work.
  // It seems the is_abstract function doesn't
  // navigate the entire set of base classes,
  // and therefore, it doesn't detect B::foo()
  //
#ifdef SWIG
  // Uncommenting this line, of course, make it works
  // %feature("notabstract") E;
#endif
  //
  struct E : C, B
  {
    E()
    {
    }    
  };
%}

