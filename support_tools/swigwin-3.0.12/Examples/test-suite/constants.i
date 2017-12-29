%module constants


%inline %{
  
  struct A {
    A(double) { }

  };

  const A b123(3.0);

%}

  

