/* This interface checks whether SWIG supports the new double angled brackets
   in the template syntax without having a space inbetween. This feature was
   introduced in new C++11 standard.
*/
%module cpp11_template_double_brackets
%inline %{
#include <map>
std::map<int,std::map<int, double>> map1;
std::map< int,std::map<int, double> > map2;

std::map<int,std::map<int, std::map<int, double>>> map3;
std::map<int,std::map<int, std::map<int, std::map<int, double>>>> map4;
%}

// Check streaming operators are still okay
%rename(ExtractionOperator) operator>>;
%rename(InsertionOperator) operator<<;

%inline %{
class ABC {
public:
  int a;
  int operator>>(ABC &) { return 0; }
  int operator<<(ABC &) { return 0; }
};

class DEF {
public:
  int a;
  int operator<<(DEF &) { return 0; }
  int operator>>(DEF &) { return 0; }
};


template<class T>
class ABC2 {
public:
  int a;

  template<typename U>
    U operator>>(ABC &);

  template<typename U>
    U operator<<(ABC &);
};
%}

// Test shifts are still working
%inline %{
int shift_init1 = 4 << 2 >> 1;
int shift_init2 = 4 >> 2 << 1 << 1 >> 2;
%}

