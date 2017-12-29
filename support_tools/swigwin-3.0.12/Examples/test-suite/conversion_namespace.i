%module conversion_namespace
%rename(toFoo) oss::Bar::operator Foo();

%inline %{ 
 namespace oss 
 { 
   struct Foo {
   };
   struct Bar {
      operator Foo () { return Foo(); }
   };
  } 
%} 

