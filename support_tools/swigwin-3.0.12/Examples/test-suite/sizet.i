%module sizet
%{
#include <vector>
%}

#ifndef SWIGCHICKEN
%include "std_common.i"
#endif

%inline
{
  size_t test1(size_t s)
  {
    return s;
  }

  std::size_t test2(std::size_t s)
  {
    return s;
  }

  const std::size_t& test3(const std::size_t& s)
  {
    return s;
  }

  const size_t& test4(const size_t& s)
  {
    return s;
  }

}

#ifdef SWIGPYTHON
%include "std_vector.i"

%template(vectors) std::vector<unsigned long>;
  
%inline 
{
  std::vector<std::size_t> testv1(std::vector<std::size_t> s)
  {
    return s;
  }

  const std::vector<std::size_t>& testv2(const std::vector<std::size_t>& s)
  {
    return s;
  }

}
#endif
