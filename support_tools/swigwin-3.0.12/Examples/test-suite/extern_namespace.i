%module extern_namespace
%inline %{

  namespace foo {
    
    extern int bar(int blah = 1);
    
  }

  extern "C" int foobar(int i) {
    return i;
  }
  
%}



%{
  int foo::bar(int blah) {
    return blah;
  }
%}
  
