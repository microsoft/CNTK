%module namespace_typedef_class

%inline %{
namespace ns {
  
  struct S1
  {
    int n;
  };

  typedef struct
  {
    int n;
  } S2;
}

%}

