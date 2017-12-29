/* Tests the use of %template with namespaces */

%module namespace_template

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) vector<int>;            /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) test2::vector<short>;   /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) test3::vector<long>;    /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) vector<test4::Integer>; /* Ruby, wrong class name */

%{
#ifdef max
#undef max
#endif
%}

%{
namespace test {
   template<typename T> T max(T a, T b) { return (a > b) ? a : b; }
   template<typename T> class vector { 
	public:
	   vector() { }
          ~vector() { }
           char * blah(T x) {
              return (char *) "vector::blah";
           }
   }; 
}

namespace test2 {
   using namespace test;
}

namespace test3 {
   using test::max;
   using test::vector;
}

using namespace test2;
namespace T4 = test;
%}

namespace test {
   template<typename T> T max(T a, T b) { return (a > b) ? a : b; }
   template<typename T> class vector { 
	public:
	   vector() { }
          ~vector() { }
           char * blah(T x) {
              return (char *) "vector::blah";
           }
   }; 
}

using namespace test;
%template(maxint) max<int>;
%template(vectorint) vector<int>;

namespace test2 {
   using namespace test;
   %template(maxshort) max<short>;
   %template(vectorshort) vector<short>;
}

namespace test3 {
   using test::max;
   using test::vector;
   %template(maxlong) max<long>;
   %template(vectorlong) vector<long>;
}

%inline %{

namespace test4 {
   using namespace test;
   typedef int Integer;
}

%}

namespace test4 {
   %template(maxInteger) max<Integer>;
   %template(vectorInteger) vector<Integer>;
}

