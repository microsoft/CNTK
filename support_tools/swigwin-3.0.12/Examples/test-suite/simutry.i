%module simutry

%include "std_vector.i"

%inline {

namespace simuPOP
{
  // some simple pop class
  template <class Type>
  struct Population {
    int m_a;
    Population(int a):m_a(a){}
  };

  // base operator, output pop.m_a
  template<class Pop>
  struct Operator
  {
    Pop m_pop;
    Operator(int a):m_pop(a){}
    virtual ~Operator()
    {
    }
    
    virtual int func() const 
    { return m_pop.m_a; }
  };

  // derived operator, output double of pop.m_a
  template<class Pop>
  struct DerivedOperator: public Operator<Pop>
  {
    DerivedOperator(int a):Operator<Pop>(a){}
    virtual int func() const 
    { return 2*this->m_pop.m_a; }
  };

}

}

#if 1
namespace simuPOP
{
  %template(population)   Population< std::pair<unsigned long,unsigned long> >;
}      

%inline 
{
  namespace simuPOP
  {
    typedef Population< std::pair<unsigned long,unsigned long> > pop;
  }
}
#else
%inline 
{
  namespace simuPOP
  {
    //  %template(population)          Population< std::pair<unsigned long,unsigned long> >;
    
    struct pop {
      int m_a;
      pop(int a):m_a(a){}
    };
  }
}
#endif


namespace simuPOP
{
 %template(baseOperator)        Operator< pop >;
 %template(derivedOperator)     DerivedOperator< pop >;
}



namespace std
{
  %template(vectorop)   vector< simuPOP::Operator<simuPOP::pop> * >;
}

%inline
{
namespace simuPOP
{
  // test function, use of a vector of Operator*
  void test( const std::vector< Operator<pop>*>& para)
  {
    for( size_t i =0; i < para.size(); ++i)
    para[i]->func();
  }
}
}


