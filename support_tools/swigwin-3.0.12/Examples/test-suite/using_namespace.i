%module(ruby_minherit="1") using_namespace

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) hi::hi0;	/* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) hi::hi1;	/* Ruby, wrong class name */

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) Hi<hello::Hello, hi::hi0>; // C#, D, Java, PHP multiple inheritance

%inline %{
  namespace hello
  {  
    struct Hello 
    {
    };

    template <class _T1, class _T2>
    struct Hi : _T1, _T2
    {
      int value1() const
      {
	return 1;
      }      

      int value2() const
      {
	return 2;
      }      
    };    
  }

  namespace hi
  {

    struct hi0
    {
    };
    
  }
%}

namespace hello
{
  %template(Hi_hi0) Hi<hello::Hello, hi::hi0>;
}


%inline %{
  namespace hi
  {
    struct hi1 : private hello::Hi< hello::Hello, hi0 >
    {
      using hello::Hi< hello::Hello, hi::hi0 >::value1;
      using hello::Hi< hello::Hello, hi0 >::value2;
    };
    
  }
  
%}


%inline {
namespace foo {
  typedef double mytype;
}

// global namespace
typedef float mytype;

using namespace foo;

struct X {
  ::mytype d;
};

}

%inline %{
namespace SpaceMan {
  typedef double SpaceManDouble;
}
using namespace ::SpaceMan; // global namespace prefix

SpaceManDouble useSpaceMan(SpaceManDouble s) { return s; }

%}

