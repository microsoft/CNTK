%module li_std_except

%include <std_except.i>

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}


%inline %{
  struct E1 : public std::exception
  {
  };

  struct E2 
  {
  };

  struct Test {
    int foo1() throw(std::bad_exception) { return 0; }
    int foo2() throw(std::logic_error) { return 0; }
    int foo3() throw(E1) { return 0; }
    int foo4() throw(E2) { return 0; }
    // all the STL exceptions...
    void throw_bad_cast()         throw(std::bad_cast)          { throw std::bad_cast(); }
    void throw_bad_exception()    throw(std::bad_exception)     { throw std::bad_exception(); }
    void throw_domain_error()     throw(std::domain_error)      { throw std::domain_error("oops"); }
    void throw_exception()        throw(std::exception)         { throw std::exception(); }
    void throw_invalid_argument() throw(std::invalid_argument)  { throw std::invalid_argument("oops"); }
    void throw_length_error()     throw(std::length_error)      { throw std::length_error("oops"); }
    void throw_logic_error()      throw(std::logic_error)       { throw std::logic_error("oops"); }
    void throw_out_of_range()     throw(std::out_of_range)      { throw std::out_of_range("oops"); }
    void throw_overflow_error()   throw(std::overflow_error)    { throw std::overflow_error("oops"); }
    void throw_range_error()      throw(std::range_error)       { throw std::range_error("oops"); }
    void throw_runtime_error()    throw(std::runtime_error)     { throw std::runtime_error("oops"); }
    void throw_underflow_error()  throw(std::underflow_error)   { throw std::underflow_error("oops"); }
  };
%}
