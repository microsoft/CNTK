%module r_legacy
%inline %{
typedef char *String;

typedef struct {
  int i;
  double d;
  char *str;
  String s;
} Obj;

Obj *getObject(int i, double d);

#include <string.h>

Obj *
getObject(int i, double d)
{

  const char *test_string = "a test string";
  Obj *obj;
  obj = (Obj *) calloc(1, sizeof(Obj));

  obj->i = i;
  obj->d = d;
  /* allocate one extra byte for the null */
  obj->str = (char *)malloc(strlen(test_string) + 1);
  strcpy(obj->str, test_string);

  return(obj);
}
%}

char *getString();
int getInt();
double getDouble();
float getFloat();
long getLong();
unsigned long getUnsignedLong();
char getChar();

extern unsigned long MyULong;

extern const double PiSquared;

#if 0
extern float *MyFloatRef;
#endif

%inline %{
#define PI 3.14159265358979
unsigned long MyULong = 20;

static float MyFloat = 1.05f;
float *MyFloatRef = &MyFloat;

const double PiSquared = PI * PI;

char *getString()
{
  return "This is a literal string";
}

int 
getInt()
{
 return 42;
}

double 
getDouble()
{
  return PI;
}

float 
getFloat()
{
  return (float)PI/2;
}

long getLong()
{
  return -321313L;
}

unsigned long 
getUnsignedLong()
{
  return 23123L;
}

char
getChar()
{
  return('A');
}
%}
