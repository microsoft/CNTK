%module(directors="1") director_constructor

%feature("director") Foo;

%inline %{
class Foo
{
public:
  int a;
  
  Foo(int i)
  {
    a=i;
  }
  
  virtual ~Foo() { }
  
  int do_test() {
    return test();
  }
  
  virtual int getit()
  {
    return a;
  }
  
  virtual void doubleit()
  {
    a = a * 2;
  }
  
  virtual int test() = 0; 
};
%}  
  


