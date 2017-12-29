%module template_default_arg_virtual_destructor

// SF bug #1296:
// virtual destructor in template class (template specification having 
// default parameter(s)) triggers the warning "illegal destructor name"

%inline %{
struct A {};

template <class X, class T = int>
  struct B 
  { 
    B(T const&) {}
    virtual ~B() {} 
  };
template <class X>
  struct B<X,int>
  { 
    B(int,int) {}   // constructor specific to this partial specialization
    virtual ~B() {} // "illegal destructor name" when ~B() is virtual
  };
%}
%template(B_AF) B<A,float>;
%template(B_A) B<A>; // this instantiation triggers the warning
