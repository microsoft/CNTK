%module member_pointer

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, badargtype2w) /* Formal argument ... is being passed extern "C" ... */
#pragma error_messages (off, wbadinit) /* Using extern "C" ... to initialize ... */
#pragma error_messages (off, wbadasg) /* Assigning extern "C" ... */
#endif
%}

%inline %{
class Shape {
public:
  Shape() {
    nshapes++;
  }
  virtual ~Shape() {
    nshapes--;
  };
  double  x, y;   
  double  *z;

  void    move(double dx, double dy);
  virtual double area(void) = 0;
  virtual double perimeter(void) = 0;
  static  int nshapes;
};

class Circle : public Shape {
private:
  double radius;
public:
  Circle(double r) : radius(r) { };
  virtual double area(void);
  virtual double perimeter(void);
};
  
class Square : public Shape {
private:
  double width;
public:
  Square(double w) : width(w) { };
  virtual double area(void);
  virtual double perimeter(void);
};

extern double do_op(Shape *s, double (Shape::*m)(void));

/* Functions that return member pointers */

extern double (Shape::*areapt())(void);
extern double (Shape::*perimeterpt())(void);

/* Global variables that are member pointers */
extern double (Shape::*areavar)(void);
extern double (Shape::*perimetervar)(void);

%}

%{
#  define SWIG_M_PI 3.14159265358979323846

/* Move the shape to a new location */
void Shape::move(double dx, double dy) {
  x += dx;
  y += dy;
}

int Shape::nshapes = 0;

double Circle::area(void) {
  return SWIG_M_PI*radius*radius;
}

double Circle::perimeter(void) {
  return 2*SWIG_M_PI*radius;
}

double Square::area(void) {
  return width*width;
}

double Square::perimeter(void) {
  return 4*width;
}

double do_op(Shape *s, double (Shape::*m)(void)) {
  return (s->*m)();
}

double (Shape::*areapt())(void) {
  return &Shape::area;
}

double (Shape::*perimeterpt())(void) {
  return &Shape::perimeter;
}

/* Member pointer variables */
double (Shape::*areavar)(void) = &Shape::area;
double (Shape::*perimetervar)(void) = &Shape::perimeter;
%}


/* Some constants */
%constant double (Shape::*AREAPT)(void) = &Shape::area;
%constant double (Shape::*PERIMPT)(void) = &Shape::perimeter;
%constant double (Shape::*NULLPT)(void) = 0;

/*
%inline %{
  struct Funktions {
    void retByRef(int & (*d)(double)) {}
  };
  void byRef(int & (Funktions::*d)(double)) {}
%}
*/

%inline %{

struct Funktions {
  int addByValue(const int &a, int b) { return a+b; }
  int * addByPointer(const int &a, int b) { static int val; val = a+b; return &val; }
  int & addByReference(const int &a, int b) { static int val; val = a+b; return val; }
};

int call1(int (Funktions::*d)(const int &, int), int a, int b) { Funktions f; return (f.*d)(a, b); }
int call2(int * (Funktions::*d)(const int &, int), int a, int b) { Funktions f; return *(f.*d)(a, b); }
int call3(int & (Funktions::*d)(const int &, int), int a, int b) { Funktions f; return (f.*d)(a, b); }
%}

%constant int (Funktions::*ADD_BY_VALUE)(const int &, int) = &Funktions::addByValue;
%constant int * (Funktions::*ADD_BY_POINTER)(const int &, int) = &Funktions::addByPointer;
%constant int & (Funktions::*ADD_BY_REFERENCE)(const int &, int) = &Funktions::addByReference;

