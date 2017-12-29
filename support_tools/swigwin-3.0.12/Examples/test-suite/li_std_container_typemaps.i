%module li_std_container_typemaps

%include stl.i
%include std_list.i
%include std_deque.i
%include std_set.i
%include std_multiset.i

%{
#include <vector>
#include <list>
#include <deque>
#include <set>

#include <iostream>
#include <iterator>
#include <algorithm>
#include <numeric>

using namespace std;
%}

%inline %{
class ClassA
{
public:
  ClassA() : a(0) {}
  ClassA(int _a) : a(_a) {}
  ClassA(const ClassA& c) : a(c.a) {}
  int a;
};

typedef ClassA* ClassAPtr;

enum _Color { RED=1, GREEN=10, YELLOW=11, BLUE=100, MAGENTA=101, CYAN=111 };
typedef enum _Color Color;

namespace std {
  template<typename T> T binaryOperation(T x, T y) {
    return static_cast<T>(x + y);
  }

  template<> bool binaryOperation(bool x, bool y) {
    return x | y;
  }

  template<> ClassAPtr binaryOperation(ClassAPtr x, ClassAPtr y) {
    if (x)
      y->a += x->a;
    return y;
  }

  template<typename SeqCont>
  struct sequence_container {
    typedef typename SeqCont::value_type value_type;

    static SeqCont ret_container(const value_type value1, const value_type value2) {
      SeqCont s;
      s.insert(s.end(), value1);
      s.insert(s.end(), value2);
      return s;
    }

    static value_type val_container(const SeqCont container) {
      return std::accumulate(container.begin(), container.end(), value_type(),
        binaryOperation<value_type>);
    }

    static value_type ref_container(const SeqCont& container) {
      return std::accumulate(container.begin(), container.end(), value_type(),
        binaryOperation<value_type>);
    }
  };

  template<typename T, class Container>
  Container ret_container(const T value1, const T value2) {
    return sequence_container<Container>::ret_container(value1, value2);
  }
  template<typename T, class Container>
  T val_container(const Container container) {
    return sequence_container<Container >::val_container(container);
  }
  template<typename T, class Container>
  T ref_container(const Container& container) {
    return sequence_container<Container >::ref_container(container);
  }
}
%}

%define %instantiate_containers_templates(TYPE...)
namespace std
{
  %template(TYPE ## _vector) std::vector<TYPE>;
  %template(TYPE ## _list) std::list<TYPE>;
  %template(TYPE ## _deque) std::deque<TYPE>;
  %template(TYPE ## _set) std::set<TYPE>;
  %template(TYPE ## _multiset) std::multiset<TYPE>;
}
%enddef

%define %instantiate_containers_functions(TYPE...)
namespace std
{
  %template(ret_ ## TYPE ## _vector) ret_container<TYPE, std::vector<TYPE> >;
  %template(val_ ## TYPE ## _vector) val_container<TYPE, std::vector<TYPE> >;
  %template(ref_ ## TYPE ## _vector) ref_container<TYPE, std::vector<TYPE> >;
  %template(ret_ ## TYPE ## _list) ret_container<TYPE, std::list<TYPE> >;
  %template(val_ ## TYPE ## _list) val_container<TYPE, std::list<TYPE> >;
  %template(ref_ ## TYPE ## _list) ref_container<TYPE, std::list<TYPE> >;
  %template(ret_ ## TYPE ## _deque) ret_container<TYPE, std::deque<TYPE> >;
  %template(val_ ## TYPE ## _deque) val_container<TYPE, std::deque<TYPE> >;
  %template(ref_ ## TYPE ## _deque) ref_container<TYPE, std::deque<TYPE> >;
  %template(ret_ ## TYPE ## _set) ret_container<TYPE, std::set<TYPE> >;
  %template(val_ ## TYPE ## _set) val_container<TYPE, std::set<TYPE> >;
  %template(ref_ ## TYPE ## _set) ref_container<TYPE, std::set<TYPE> >;
  %template(ret_ ## TYPE ## _multiset) ret_container<TYPE, std::multiset<TYPE> >;
  %template(val_ ## TYPE ## _multiset) val_container<TYPE, std::multiset<TYPE> >;
  %template(ref_ ## TYPE ## _multiset) ref_container<TYPE, std::multiset<TYPE> >;
}
%enddef

%define %instantiate_containers_templates_and_functions(TYPE...)
  %instantiate_containers_templates(TYPE);
  %instantiate_containers_functions(TYPE);
%enddef

%instantiate_containers_templates_and_functions(int);
%instantiate_containers_templates_and_functions(double);
%instantiate_containers_templates_and_functions(float);
%instantiate_containers_templates_and_functions(bool);
%instantiate_containers_templates_and_functions(string);
%instantiate_containers_templates_and_functions(ClassAPtr);
