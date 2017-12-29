%module defvalue_constructor
%inline %{

namespace Foo { 
 
    class Bar {}; 
 
    class Baz { 
      public: 
        Baz(Bar b = Bar()) {}
    }; 
} 

%}
