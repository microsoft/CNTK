/* This testcase shows how to replace std_initializer_list with std_vector. */

%module cpp11_initializer_list_extend

%ignore Container::Container(std::initializer_list<int>);
%include <std_vector.i>
%template(VectorInt) std::vector<int>;

%extend Container {
  Container(const std::vector<int> &elements) {
    Container *c = new Container();
    for (int element : elements)
      c->push_back(element);
    return c;
  }
}


%inline %{
#include <initializer_list>

class Container {
public:
  Container(std::initializer_list<int>) {}
  Container() {}
  void push_back(const int&) {}
};
%}

