%module conversion
%rename(toFoo) Bar::operator Foo();

%inline %{ 
   struct Foo {
   };
   struct Bar {
      operator Foo () { return Foo(); }
   };
%} 

