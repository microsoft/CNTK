/* This testcase tests nested derived classes.
This was reported in bug #909389 */

%module derived_nested

%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) BB::CC;
%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) BB::DD;
%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) BB::EE;
%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) BB::FF;

%inline %{

class A {
public:
  int x;
};
class B {
  class C { int y; }; //generates a warning
  class D : public A { int z; }; //ok
};

struct BB {
  class CC { int y; };
  class DD : public A { int z; };
  struct EE : public A { int z; };
  struct FF : public A { int z; } ff_instance; // Bug 1960977
  void useEE(const EE& e) {}
};
%}

