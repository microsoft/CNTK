%module template_ns_scope
// Tests a scoping bug reported by Marcelo Matus.

%inline %{
  namespace hi {
    enum Hello { Hi, Hola };
 
    template <Hello h>
    struct A
    {
    public:
      A() {}    // *** Here, the const. breaks swig ***
                // *** swig  works without it     ***
    };
 
    namespace hello
    {
      template <Hello H>
      struct B : A<H>
      {
        int say_hi() { return 0; }
      };
    }
  }
 
%}
namespace hi
{
  %template(A_Hi) A<Hi>;
  namespace hello
  {
    %template(B_Hi) B<Hi>;
  }
}                                           




