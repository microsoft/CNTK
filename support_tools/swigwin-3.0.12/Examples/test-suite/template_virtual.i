%module template_virtual

// Submitted by Marcelo Matus  
// assertion emmitted with templates + derivation + pure virtual member
// allocate.cxx:47: int Allocate::is_abstract_inherit(Node*, Node*):
// Assertion `dn' failed.
 
%inline %{
 
    template <class T>
    class A
    {
    public:
      virtual ~A() { }

      virtual void say_hi() = 0; // only fails with pure virtual methods
 
      virtual void say_hello() {} // this works fine
 
    protected:
      A() { }  // defined protected as swig generates constructor by default
    };
 
    template <class T>
    class B : public A<T>
    {
    protected:
      B() { } // defined protected as swig generates constructor by default
    };
 
%}
 
%template(A_int) A<int>;
%template(B_int) B<int>;  // !!!! it crashes right here !!!!!                       
