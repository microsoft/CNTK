%module member_funcptr_galore

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, badargtype2w) /* Formal argument ... is being passed extern "C" ... */
#pragma error_messages (off, wbadinit) /* Using extern "C" ... to initialize ... */
#pragma error_messages (off, wbadasg) /* Assigning extern "C" ... */
#endif
%}

%inline %{

namespace FunkSpace {
struct Funktions {
  int addByValue(const int &a, int b) { return a+b; }
  int * addByPointer(const int &a, int b) { static int val; val = a+b; return &val; }
  int & addByReference(const int &a, int b) { static int val; val = a+b; return val; }
};
}

template <typename T> struct Thing {};
namespace Space {
class Shape {
public:
  double  x, y;   
  double  *z;

  void    move(double dx, double dy);
  virtual double area(Shape &ref, int & (FunkSpace::Funktions::*d)(const int &, int)) { return 0.0; }
  virtual double abc(Thing<short> ts, Thing< const Space::Shape * > tda[]) { return 0.0; }
  virtual ~Shape() {}
};
}

extern double do_op(Space::Shape *s, double (Space::Shape::*m)(void));

/* Functions that return member pointers */

extern double (Space::Shape::*areapt())(Space::Shape &, int & (FunkSpace::Funktions::*)(const int &, int));
extern double (Space::Shape::*abcpt())(Thing<short>, Thing< const Space::Shape * > tda[]);

/* Global variables that are member pointers */
extern double (Space::Shape::*areavar)(Space::Shape &, int & (FunkSpace::Funktions::*)(const int &, int));
extern double (Space::Shape::*abcvar)(Thing<short>, Thing< const Space::Shape * >[]);

%}

%{
void Space::Shape::move(double dx, double dy) {
  x += dx;
  y += dy;
}

double do_op(Space::Shape *s, double (Space::Shape::*m)(void)) {
  return (s->*m)();
}

double (Space::Shape::*areapt(Space::Shape &ref, int & (FunkSpace::Funktions::*d)(const int &, int)))(Space::Shape &, int & (FunkSpace::Funktions::*d)(const int &, int)) {
  return &Space::Shape::area;
}

double (Space::Shape::*areapt())(Space::Shape &, int & (FunkSpace::Funktions::*)(const int &, int)) {
  return 0;
}

double (Space::Shape::*abcpt())(Thing<short>, Thing< const Space::Shape * >[]) {
  return &Space::Shape::abc;
}

/* Member pointer variables */
double (Space::Shape::*areavar)(Space::Shape &, int & (FunkSpace::Funktions::*)(const int &, int)) = &Space::Shape::area;
double (Space::Shape::*abcvar)(Thing<short>, Thing< const Space::Shape * >[]) = &Space::Shape::abc;
%}


/* Some constants */
%constant double (Space::Shape::*AREAPT)(Space::Shape &, int & (FunkSpace::Funktions::*)(const int &, int)) = &Space::Shape::area;
%constant double (Space::Shape::*PERIMPT)(Thing<short>, Thing< const Space::Shape * >[]) = &Space::Shape::abc;
%constant double (Space::Shape::*NULLPT)(void) = 0;

%inline %{

int call1(int (FunkSpace::Funktions::*d)(const int &, int), int a, int b) { FunkSpace::Funktions f; return (f.*d)(a, b); }
int call2(int * (FunkSpace::Funktions::*d)(const int &, int), int a, int b) { FunkSpace::Funktions f; return *(f.*d)(a, b); }
int call3(int & (FunkSpace::Funktions::*d)(const int &, int), int a, int b) { FunkSpace::Funktions f; return (f.*d)(a, b); }
%}

%constant int (FunkSpace::Funktions::*ADD_BY_VALUE)(const int &, int) = &FunkSpace::Funktions::addByValue;
%constant int * (FunkSpace::Funktions::*ADD_BY_POINTER)(const int &, int) = &FunkSpace::Funktions::addByPointer;
%constant int & (FunkSpace::Funktions::*ADD_BY_REFERENCE)(const int &, int) = &FunkSpace::Funktions::addByReference;

%inline %{
// parameter that is a member pointer containing a function ptr, urgh :)
int unreal1(double (Space::Shape::*memptr)(Space::Shape &, int & (FunkSpace::Funktions::*)(const int &, int))) { return 0; }
int unreal2(double (Space::Shape::*memptr)(Thing<short>)) { return 0; }
%}
