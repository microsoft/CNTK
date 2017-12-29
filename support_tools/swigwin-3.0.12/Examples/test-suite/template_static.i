%module template_static

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) foo<int>;    /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) foo<double>; /* Ruby, wrong class name */

%inline %{
template<class T> class foo {
public:
    static int test;
};
template<class T> int foo<T>::test = 0;
%}

%template(foo_i) foo<int>;
%template(foo_d) foo<double>;


%inline %{
namespace toto {
  class Foo {
  public:
      template<class T>
      static double bar(int i) {
	return 1.0;
      }

    private:
      int i;
  };
} 
%}

%template(bar_double) toto::Foo::bar<double>; 
