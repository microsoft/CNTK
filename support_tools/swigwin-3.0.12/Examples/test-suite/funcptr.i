%module funcptr

/*
 Complicated one that should defeat just reading , to find
 the number of arguments expected in the function pointer.
extern void do(int (*op)(int (*i)(double, double), int j)); 
*/

%inline %{
typedef double (*DistFun)(double* data, int r, int c, int i, int j, void *xdata);

void distance(double *data, int *dim, DistFun fun,  double *output) {
}

typedef int (*Operator)(int i,int j);

int do_op(int a, int b, int (*op)(int,int)) {
  return (*op)(a,b);
}

int add(int a, int b) {
  return a+b;
}

int subtract(int a, int b) {
  return a-b;
}

int multiply(int a, int b) {
  return a*b;
}

int *nowt() { 
  return 0;
}

int *nowt2(void) { 
  return 0;
}

struct MyStruct { int i; };
typedef struct MyStruct * MyStructPtr;

MyStructPtr mystructptr() { 
  return 0;
}

typedef int * Integer;

int (*funcvar)(int,int) = add;
int * (*funcvar2)() = nowt;
int * (*funcvar3)(void) = nowt2;
Integer (*funcvar4)() = nowt;
MyStructPtr (*funcvar5)() = mystructptr;

void (*pfunc0)();
int (*pfuncA)();
void (*pfunc1)(int);
void (*pfunc2)(int, double);
%}
