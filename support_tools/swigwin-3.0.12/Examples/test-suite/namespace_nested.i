%module namespace_nested

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) hello::hi::hi0;	/* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) oss::hi1<hello::Hi0 >;	/* Ruby, wrong class name */

%inline %{
  namespace hello
  {  
    namespace hi
    {      
      struct hi0
      {
      };      
    }

    template < class T1 >
    struct Hi : T1
    {
    };
  }

%}

namespace hello 
{
  %template(Hi_hi0) Hi<hi::hi0>;
}



%inline %{

  namespace hello
  {
    //
    // This works 
    //
    // typedef Hi<hello::hi::hi0> Hi0;
    
    //
    // This doesn't work
    //
    typedef Hi<hi::hi0> Hi0;
  }
  
  
  namespace oss
  {
    template <class T1>
    struct hi1 : T1
    {
    };

    typedef hello::Hi<hello::hi::hi0> h0;
  }
  
%}

namespace oss
{
  %template(hi1_hi0) hi1<hello::Hi0 >;
}


%rename(MyFoo) geos::algorithm::Foo;

%inline 
{
  namespace geos {
    namespace algorithm {
      class Foo 
      {
      };
    }
    
    namespace planargraph { // geos.planargraph
      namespace algorithm { // geos.planargraph.algorithm
	class Bar {
	};
      }
      namespace algorithm { // geos.planargraph.algorithm

	class Foo {
	public:
	  typedef int size_type;
	};
      }
      namespace algorithm { // geos.planargraph.algorithm

	class ConnectedSubgraphFinder : public Foo {
	public:
	  ConnectedSubgraphFinder(size_type)
	  {
	  }
	  
	};
      }
    }
  }
}

