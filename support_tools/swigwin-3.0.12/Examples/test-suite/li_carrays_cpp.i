%module li_carrays_cpp

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) doubleArray; /* Ruby, wrong class name */

%include <carrays.i>

%array_functions(int,intArray);
%array_class(double, doubleArray);
%array_class(short, shortArray);

%inline %{
typedef struct {
  int x;
  int y;
} XY;
XY globalXYArray[3];

typedef struct {
  int a;
  int b;
} AB;

AB globalABArray[3];
%}

// Note that struct XY { ... }; gives compiler error for C when using %array_class or %array_functions, but is okay in C++
%array_class(XY, XYArray)
%array_functions(AB, ABArray)

%inline %{
short sum_array(short x[5]) {
  short sum = 0;
  int i;
  for (i=0; i<5; i++) {
    sum = sum + x[i];
  }
  return sum;
}
%}
