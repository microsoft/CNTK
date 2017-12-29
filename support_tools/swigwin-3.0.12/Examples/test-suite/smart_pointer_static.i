%module smart_pointer_static
%inline %{
class Foo2;

class  MyHandle_Foo2 { 
 public: 
  Foo2 * operator -> (); 
}; 

class Foo2 { 
 public: 
  virtual ~Foo2() {}
  virtual int sum(int i, int j) { return i+j; }
  static int sum(int i, int j, int k); 
}; 

%}

%{
int Foo2::sum(int i, int j, int k) {
  return i+j+k;
}
Foo2 * MyHandle_Foo2::operator -> () { return new Foo2(); }
%}
