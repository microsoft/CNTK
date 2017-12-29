%module typedef_reference

%include cpointer.i
%pointer_functions(int, intp);

%inline %{
  typedef int & IntRef;
  int somefunc(IntRef i) { return i; }
  int otherfunc(int &i) { return i; }
%}
