%module compactdefaultargs

// compactdefaultargs off by default

// The following should compile with compactdefaultargs off
%inline %{
class Defaults1 {
  static const int PRIVATE_DEFAULT = -1;
public:
  static const double PUBLIC_DEFAULT;
  Defaults1(int a = PRIVATE_DEFAULT) {}
  double ret(double d = PUBLIC_DEFAULT) { return d; }
};
%}

%{
const double Defaults1::PUBLIC_DEFAULT = -1.0;
%}

// compactdefaultargs now on by default
%feature("compactdefaultargs");

// Turn compactdefaultargs off for the constructor which can't work with this feature
%feature("compactdefaultargs", "0") Defaults2(int a = PRIVATE_DEFAULT);

%inline %{
class Defaults2 {
  static const int PRIVATE_DEFAULT = -1;
public:
  static const double PUBLIC_DEFAULT;
  Defaults2(int a = PRIVATE_DEFAULT) {}
  double ret(double d = PUBLIC_DEFAULT) { return d; }
  double nodefault(int x) { return x; }
};
%}

%{
const double Defaults2::PUBLIC_DEFAULT = -1.0;
%}
