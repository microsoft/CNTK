%module virtual_derivation

 /*

 Try to add to your favorite language a runtime test like
 this:
 
 b = B(3)
 if (b.get_a() != b.get_b()):
     print "something is wrong"


 The test runs fine with python, but not with ruby.
 
 */

%inline %{

  struct A 
  {
    ~A()
    {
    }
    
    int m_a;
    
    A(int a) :m_a(a)
    {
    }
    
    int get_a()
    {
      return m_a;
    }
    
  };
  
  struct B : virtual A
  {
    B(int a): A(a)
    {
    }
    
    int get_b()
    {
      return get_a();
    }

    // in ruby, get_a() returns trash if called from b, unless is
    // wrapped with the previous get_b or using the 'using'
    // declaration:
    // using A::get_a;
  };




  class IndexReader{
  public:
    virtual void norms() = 0;
  };

  class MultiReader : public IndexReader {
  protected:
    MultiReader();
  };
%}
