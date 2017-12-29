%module namespace_enum

%inline %{

namespace Foo {
   enum Swig {
       LAGER,
       STOUT,
       ALE
    };

   class Bar {
   public:
        enum Speed {
            SLOW,
            FAST
        };
   };
}

%}

       
   