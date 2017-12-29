/* This interface file checks whether the SWIG parses the throw
   directive in combination with the const directive.  Bug reported by
   Scott B. Drummonds, 08 June 2001.  
*/

%module cplusplus_throw

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%nodefaultctor;

%inline %{

class Foo { };

class Bar {
public:
  void baz() const { };
  void foo() throw (Foo) { };
  void bazfoo() const throw (int) { };
};

%}

