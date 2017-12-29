%module xxx

%inline %{
struct A5;
int A6;
template<typename T> struct A7
{
};
template<typename T> struct A8 
{
};

struct A0 
: 
  public A1
  ,
  A2, 
  private A3
  ,
  private A4
  ,
  A5
  ,
  A6
  ,
  A7<int>
  ,
  protected A8<double>
{    
};

struct A1
{
};

class B1 {};

class B0 :
  B1,
  B2<int>
{
};

struct Recursive : Recursive
{
};
%}


template <typename T> class Base {};
%template() Base<int>;
class Derived : public Base<int> {};
class Derived2 : public Base<double> {};
%template(BaseDouble) Base<double>;

