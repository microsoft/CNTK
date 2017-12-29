%module template_typemaps_typedef
// Similar to template_typedef_class_template
// Testing typemaps of a typedef of a nested class in a template and where the template uses default parameters

%inline %{
namespace Standard {
  template <class T, class U > struct Pair {
    T first;
    U second;
  };
}
%}

%{
namespace Standard {
template<class Key, class T, class J = int> class Multimap {
  public:
    typedef Key key_type;
    typedef T mapped_type;

    class iterator {
    public:
      mapped_type mm;
      iterator(mapped_type m = mapped_type()) : mm(m) {}
    };

    mapped_type typemap_test(Standard::Pair<iterator,iterator> pp) { return pp.second.mm; }
    Standard::Pair<iterator,iterator>* make_dummy_pair() { return new Standard::Pair<iterator, iterator>(); }
  };
}
%}

namespace Standard {
template<class Key, class T, class J = int> class Multimap {
  public:
    typedef Key key_type;
    typedef T mapped_type;

    class iterator;

    %typemap(in) Standard::Pair<iterator,iterator> "$1 = default_general< Key, T >();"
    mapped_type typemap_test(Standard::Pair<iterator,iterator> pii1);
    Standard::Pair<iterator,iterator>* make_dummy_pair();
  };
}

// specialization
namespace Standard {
template<> class Multimap<A, int> {
  public:
    typedef A key_type;
    typedef int mapped_type;

    class iterator;

    // Note uses a different function to the non-specialized version
    %typemap(in) Standard::Pair<iterator,iterator> "$1 = default_A_int< A, int >();"
    mapped_type typemap_test(Standard::Pair<iterator,iterator> pii2);
    Standard::Pair<iterator,iterator>* make_dummy_pair();
  };
}

%inline %{
struct A {
    int val;
    A(int v = 0): val(v) {}
};
%}

%{
// For < int, A >
template<typename Key, typename T> Standard::Pair< typename Standard::Multimap< Key, T >::iterator, typename Standard::Multimap< Key, T >::iterator > default_general() {
  Standard::Pair< typename Standard::Multimap< Key, T >::iterator, typename Standard::Multimap< Key, T >::iterator > default_value;
  default_value.second.mm = A(1234);
  return default_value;
}
// For < A, int >
template<typename Key, typename T> Standard::Pair< typename Standard::Multimap< Key, T >::iterator, typename Standard::Multimap< Key, T >::iterator > default_A_int() {
  Standard::Pair< typename Standard::Multimap< Key, T >::iterator, typename Standard::Multimap< Key, T >::iterator > default_value;
  default_value.second.mm = 4321;
  return default_value;
}
%}

%inline %{
typedef A AA;
namespace Space {
  typedef AA AB;
}
%}

%template(PairIntA) Standard::Pair<int, A>;
%template(MultimapIntA) Standard::Multimap<int, A>;

%template(PairAInt) Standard::Pair<A, int>;
%template(MultimapAInt) Standard::Multimap<A, int>;

%inline %{

// Extend the test with some typedefs in the template parameters
Standard::Multimap< int, AA      >::mapped_type typedef_test1(Standard::Pair< Standard::Multimap< int, AA      >::iterator, Standard::Multimap< int, AA      >::iterator > pp) { return pp.second.mm; }
Standard::Multimap< int, A       >::mapped_type typedef_test2(Standard::Pair< Standard::Multimap< int, A       >::iterator, Standard::Multimap< int, A       >::iterator > pp) { return pp.second.mm; }
Standard::Multimap< int, AA, int >::mapped_type typedef_test3(Standard::Pair< Standard::Multimap< int, AA, int >::iterator, Standard::Multimap< int, AA, int >::iterator > pp) { return pp.second.mm; }
Standard::Multimap< int, A , int >::mapped_type typedef_test4(Standard::Pair< Standard::Multimap< int, A , int >::iterator, Standard::Multimap< int, A , int >::iterator > pp) { return pp.second.mm; }
using namespace Space;
Standard::Multimap< int, AB      >::mapped_type typedef_test5(Standard::Pair< Standard::Multimap< int, AB      >::iterator, Standard::Multimap< int, AB      >::iterator > pp) { return pp.second.mm; }
Standard::Multimap< int, AB, int >::mapped_type typedef_test6(Standard::Pair< Standard::Multimap< int, AB, int >::iterator, Standard::Multimap< int, AB, int >::iterator > pp) { return pp.second.mm; }
%}

