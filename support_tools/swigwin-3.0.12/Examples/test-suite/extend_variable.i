%module extend_variable

// Tests %extend for variables

%inline %{
class ExtendMe {
  double var;
public:
  ExtendMe() : var(0.0) {}
  bool get(double &d) {
    d = var;
    return true;
  }
  bool set(const double &d) {
    var = d;
    return true;
  }
};
%}

%extend ExtendMe {
  double ExtendVar;
};

%{
// If possible, all language modules should use this naming format for consistency
void ExtendMe_ExtendVar_set(ExtendMe *thisptr, double value) {
  thisptr->set(value);
}
double ExtendMe_ExtendVar_get(ExtendMe *thisptr) {
  double value = 0;
  thisptr->get(value);
  return value;
}
%}


%{
  class Foo 
  {
  };
%}

#if SWIGJAVA
%javaconst(1) AllBarOne;
#elif SWIGCSHARP
%csconst(1) AllBarOne;
#endif


class Foo {
  public:
    %extend {
        static const int Bar = 42;
        static const int AllBarOne = 4422;
        static const int StaticConstInt;
        static int StaticInt;
    }
}; 
  
%{
  int globalVariable = 1111;

  void Foo_StaticInt_set(int value) {
    globalVariable = value;
  }

  int Foo_StaticInt_get() {
    return globalVariable;
  }

  int Foo_StaticConstInt_get() {
    static int var = 2222;
    return var;
  }
%}

%inline {
  namespace ns1 
  {
    struct Bar
    {
    }
    ;
  }
}

%{
  int ns1_Bar_x_get(ns1::Bar *self) {
    return 1;
  }

  void ns1_Bar_x_set(ns1::Bar *self, int x) {
  }
%}
  
%extend ns1::Bar 
{
  int x;
}


  
