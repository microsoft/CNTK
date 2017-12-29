#ifndef subdir1_hello_i_
#define subdir1_hello_i_

%{
typedef int Integer;
%}

%inline %{

  struct A
  {  
    int aa;
  };

  Integer importtest1(Integer i) {
    return i + 10;
  }

%}
  



#endif //subdir1_hello_i_
