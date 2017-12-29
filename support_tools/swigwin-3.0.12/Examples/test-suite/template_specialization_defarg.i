%module template_specialization_defarg

%inline %{

  template <class A, class B = double>
    struct C
    {
    };
  
  
  template <class BB>
    struct C<int , BB> 
    {
      int hi()
      {
	return 0;
      }

      C(int a)
      {
      }
      
    };
  

  template <class BB>
    struct C<double , BB> 
    {
      int hello()
      {
	return 0;
      }
      
      C(double a)
      {
      }
      
    };

  template <class T>
    struct Alloc 
    {
    };
  

  template <class T, class A = double >
    struct D
    {
      D(int){}
    };


  template <>
    struct D<double>
    {
      D(){}
      int foo() { return 0; }
    };
  
      
  
  template <class T, class A = Alloc<T> >
    struct Vector
    {
      Vector(int){}
    };


  template <>
    struct Vector<double>
    {
      Vector(){}
      int foo() { return 0; }
    };
  
      
%}


//
// This works fine
//
%template(C_i) C<int, double>;

//
// This one fails
//
%template(C_dd) C<double,double>;
%template(C_d) C<double>;

%template(D_i) D<int>;
%template(D_d) D<double>;

%template(Vector_i) Vector<int>;
%template(Vector_d) Vector<double, Alloc<double> >;
