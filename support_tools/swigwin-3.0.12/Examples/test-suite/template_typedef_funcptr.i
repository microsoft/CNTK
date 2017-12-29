%module template_typedef_funcptr

//Bug #1832613

#if !defined(SWIGR)
// R Swig fails on this test.  Because it tries to return a nil SEXP in
// an error

%include <std_string.i>

%inline %{

#include <string>

template<typename T> class Ptr {};
 
class MCContract {};
typedef Ptr<MCContract> MCContractPtr;
%}

%template() Ptr<MCContract>;

%inline %{
template <class Contract, typename ContractID, typename CallbackType>
class ContractFactory 
{
  public:
    static ContractFactory<Contract,ContractID,CallbackType> &getInstance() {
      static ContractFactory<Contract, ContractID, CallbackType> instance;
      return instance;
    }
};
/**
 * CreateXXContractCallback is a pointer to a function taking no arguments and 
 * returning a pointer to an XXContract. 
 */
typedef MCContractPtr (*CreateMCContractCallback)();
%}


//Get around it by changing this:
%template(MCContractFactory) ContractFactory<MCContract, std::string, CreateMCContractCallback>;

//to a form which expands the typedef:
//%template(MCContractFactory) ContractFactory<MCContract, std::string, Ptr<MCContract>(*)()>;

%inline %{
typedef MCContractPtr* ContractPtrPtr;
%}
// Plain pointers were also causing problems...
%template(MCContractFactory2) ContractFactory<MCContract, std::string, ContractPtrPtr>;

#endif
