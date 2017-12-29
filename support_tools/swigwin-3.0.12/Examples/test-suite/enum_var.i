%module enum_var

%inline %{

enum Fruit { APPLE, PEAR };
Fruit test;

%}
