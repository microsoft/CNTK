%module namespace_virtual_method

%warnfilter(515);


%inline %{

namespace A {
   namespace B {
       class Foo;
   }
   namespace C {
       class Foo {
	public:
           Foo() { };
           virtual ~Foo() { };
           virtual int bar(const A::B::Foo &x) = 0;
       };
   }
}

namespace A {
   namespace C {
        class Spam : public Foo {
	public:
            Spam() { }
            virtual ~Spam() { }
            virtual int bar(const B::Foo &x) { return 1; }
        };
   }
}

%}

%{
namespace A {
   namespace B {
         class Foo { };
   }
}
%}
