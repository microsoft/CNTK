%module extend_template_ns
%inline %{ 
namespace oss 
{ 
  enum Test {One, Two}; 
} 
%} 
 
namespace oss { 
   %extend Foo<One> {           //************ this doesn't  work 
     int test1(int x) { return x; } 
   };
} 
 
%extend oss::Foo<oss::One> {  //******** this works 
int test2(int x) { return x; } 
}; 
 
%inline %{ 
namespace oss 
{ 
  template <Test> 
  struct Foo { 
  }; 
 } 
%} 
 
namespace oss 
{ 
%template(Foo_One) Foo<One>; 
} 
 
