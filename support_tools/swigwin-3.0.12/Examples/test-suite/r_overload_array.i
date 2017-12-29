%module r_overload_array

%include <stl.i>
// this tests the situation in which there is a scalar function
// corresponding with a vector one

%inline %{
class Foo {
public:
  double bar(double w) {return w;};
  double bar(double *w) {return w[0];}
  double bar(std::vector<double> w) {return w[0];}

  int bar_int(int w) {return w;}
  int bar_int(int *w) {return w[0];}
  int bar_int(std::vector<int> w) {return w[0];}
};
%}



