%module extend_template
%module xxx // should be ignored
namespace oss { // this doesn't 
 %extend Foo<0> { 
    int test1(int x) { return x; }
 } 
} 
 
%extend oss::Foo<0> { // this doesn't 
 int test2(int x) { return x; }
};

 
%inline %{ 
 namespace oss 
 { 
   template <int> 
   struct Foo { 
   }; 
  } 
%} 
 
namespace oss 
{ 
 %template(Foo_0) Foo<0>; 
} 
