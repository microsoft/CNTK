%module constructor_value
%inline %{

class Foo { 
public: 
Foo(int a) {}; 
}; 

class Bar { 
public: 
Bar(Foo ci) {} 
}; 

%}

