%module typedef_class

%inline %{ 
class RealA 
{ 
 public: 
   int a; 
}; 
 
class B 
{ 
 public: 
   typedef RealA A2; 
   int testA (const A2& a) {return a.a;} 
}; 

namespace Space {
  typedef class AAA {
  public:
    AAA() {}
  } BBB;
}

typedef class AA {
public:
  AA() {}
  AA(int x) {}
  int aa_var;
  int *aa_method(double d) { return 0; }
  static int *aa_static_method(bool b) { return 0; }
} BB;
%} 
