/* This interface tests whether SWIG supports the new "constexpr" keyword
   introduced by C++11.
*/
%module cpp11_constexpr

%inline %{
constexpr int AAA = 10;
constexpr const int BBB = 20;
constexpr int CCC() { return 30; }
constexpr const int DDD() { return 40; }

constexpr int XXX() { return 10; }
constexpr int YYY = XXX() + 100;

struct ConstExpressions {
  static constexpr const int JJJ = 100;
  static constexpr int KKK = 200;
  static const int LLL = 300;
  constexpr int MMM() { return 400; }
  constexpr const int NNN() { return 500; }
  // Regression tests for support added in SWIG 3.0.4:
  static constexpr const int JJJ1 = 101;
  constexpr static int KKK1 = 201;
  // Regression tests for https://github.com/swig/swig/issues/284 :
  explicit constexpr ConstExpressions(int) { }
  constexpr explicit ConstExpressions(double) { }
};
%}

%{
int Array10[AAA];
int Array20[BBB];
int Array30[CCC()];
int Array40[DDD()];
int Array100[ConstExpressions::JJJ];
int Array200[ConstExpressions::KKK];
int Array300[ConstExpressions::LLL];
//int Array400[ConstExpressions::MMM()];
//int Array500[ConstExpressions::NNN()];
%}
