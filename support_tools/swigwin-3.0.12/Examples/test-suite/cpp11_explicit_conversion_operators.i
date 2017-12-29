/* This interface checks whether SWIG correctly compiles the new
   explicit conversion operators feature introduced in C++11.
*/
%module cpp11_explicit_conversion_operators

%warnfilter(SWIGWARN_LANG_IDENTIFIER) Testable::operator bool;
%rename(AsInteger) Testable::operator int;

%inline %{

class U {
public:
  int u;
};

class V {
public:
  int v;
};

class TestClass {
public:
  //implicit converting constructor
  TestClass( U const &val ) { t=val.u; }
  // explicit constructor
  explicit TestClass( V const &val ) { t=val.v; }

  int t;
};

struct Testable {
  // explicit conversion operator
  explicit operator bool() const {
    return false;
  }
  explicit operator int() {
    return 42;
  }
};
%}

