%module r_copy_struct

%feature("opaque", "yes")  B;
%feature("opaque", "yes")  C;

%inline %{
struct A {
  int i;
  unsigned int ui;
  double d;
  char* str;
  int **tmp;
};

struct A getA();
struct A* getARef();

typedef struct  {
  int invisible;
} B;

struct C {
 int invisible;
 double blind;
};

typedef B C;

B* getBRef();
struct C* getCRef();

C* getCCRef();

typedef struct 
{
 int x;
 double u;
} D;

struct A 
getA()
{
  struct A a;

  a.i = 10;
  a.d = 3.1415;

  return a;
}

static struct A fixed = {20, 3, 42.0, 0, 0};

struct A *
getARef()
{
  return(&fixed);
}


static B bb = {101};

B*
getBRef()
{
  return(&bb);
}

struct C cc = {201, 3.14159};
struct C *
getCRef()
{
  return(&cc);
}


C*
getCCRef()
{
  return(&bb);
}

D
bar()
{ D a;
 a.x = 1;
 a.u = 0;
 return(a);
}

%}



