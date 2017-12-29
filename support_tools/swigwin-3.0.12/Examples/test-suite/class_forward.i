%module class_forward

%inline %{
struct A { 
   class B;
};
class C : public A {
};
%}

