/* File : example.i */
%module my_example

enum color { RED=10, BLUE, GREEN };

class Foo {
 public:
  Foo() { }
  enum speed { IMPULSE, WARP, LUDICROUS };
  void enum_test(speed s);
};

void enum_test(color c, Foo::speed s);



%include pointer.i

/* Next we'll use some typemaps */

%include typemaps.i

%typemap(out) int * {
    WHATEVER  MAKES YOU HAPPY AS RESULT
}

%typemap(in) int * {
    WHATEVER  MAKES YOU HAPPY AS PARAMETER
}

%pragma(xml) DEBUG="false";

extern  int *  my_gcd(const char * x, int * y[], int * r, int (*op)(int,int)) const;
extern double my_foo;
void my_void();
my_empty();

const double my_dutch = 1.0;

union my_union
{
    int my_iii;
    char my_ccc;
};

struct my_struct
{
public:
    virtual ~my_struct();
  int my_foo();
protected:
  int my_bar;
  double  x, y;   
  virtual double area() = 0;
  static  int nshapes;
};

class my_class : public my_struct, public my_union
{
public:
    my_class( char c );
private:
    ~my_class();
    virtual const int *  my_func( my_class , char * * x, int y[], const int & r) const;
    double my_foo[128];
    const my_int i;
};

typedef int my_int;
