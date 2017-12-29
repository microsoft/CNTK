%module template_typedef_class_template

%inline %{
namespace Standard {
  template <class T, class U > struct Pair {
    T first;
    U second;
  };
}
%}

// previously these typemaps were erroneously being used as iterator was not correctly scoped in Multimap
%typemap(out) Standard::Pair<Standard::iterator, Standard::iterator> "_this_will_not_compile_iterator_"
%typemap(out) Standard::Pair<Standard::const_iterator, Standard::const_iterator> "_this_will_not_compile_const_iterator_"

%{
namespace Standard {
template<class Key, class T> class Multimap {
  public:
    typedef Key key_type;
    typedef T mapped_type;

    class iterator {};
    class const_iterator {};

    // test usage of a typedef of a nested class in a template
    Standard::Pair<iterator,iterator> equal_range_1(const key_type& kt1) { return Standard::Pair<iterator,iterator>(); }
    Standard::Pair<const_iterator,const_iterator> equal_range_2(const key_type& kt2) const { return Standard::Pair<const_iterator,const_iterator>(); }
  };
}
%}

namespace Standard {
template<class Key, class T> class Multimap {
  public:
    typedef Key key_type;
    typedef T mapped_type;

    class iterator;
    class const_iterator;

    // test usage of a typedef of a nested class in a template
    Standard::Pair<iterator,iterator> equal_range_1(const key_type& kt1) {}
    Standard::Pair<const_iterator,const_iterator> equal_range_2(const key_type& kt2) const {}
  };
}

%inline %{
struct A {
    int val;
    A(int v = 0): val(v) {}
};
%}

%template(PairA) Standard::Pair<int, A*>;
%template(MultimapA) Standard::Multimap<int, A*>;

