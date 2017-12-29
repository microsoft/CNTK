%module template_using

%inline 
{
  
namespace foo {
  template<typename T> class Foo { };
  template<typename T> T maxk(T a, T b) { return a > b ? a : b; }
}
using foo::maxk;
 
}

%template(maxint)   foo::maxk<int>;   
%template(Foofloat) foo::Foo<float>;
%template(maxfloat) maxk<float>;    

