%module(directors="1") director_redefined

 /*
   This example generates two 'get_val' virtual members in the
   director, and since they are equivalent, the compilation fails.
 */

%feature("director")  B;

%inline 
{
  typedef int Int;

  struct A
  {
    virtual ~A()
    {
    }

    virtual int get_val(Int a)
    {
      return 0;
    }

    virtual int get_rval(const Int& a)
    {
      return 0;
    }
    
  };
  
  struct B : A
  {
    int get_val(int a)
    {
      return 1;
    }    

    int get_rval(const int& a)
    {
      return 1;
    }    

    const int& get_rrval(const int& a)
    {
      return a;
    }    
  };  
}

