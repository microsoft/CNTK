%module template_ns3

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) foo::bar<int>; /* Ruby, wrong class name */

%inline %{
namespace foo {
   typedef int Integer;

   template<class T> class bar {
   public:
       Integer test(Integer x) { return x; }
   };

}
%}

%template(bari) foo::bar<int>;

   
    

