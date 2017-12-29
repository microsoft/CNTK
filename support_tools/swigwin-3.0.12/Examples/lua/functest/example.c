/* File : example.c */

/* A global variable */
double Foo = 3.0;

int add1(int x, int y)
{
   return x+y;
}

void add2(int x, int *y, int *z)
{
   *z = x+*y;
}

int add3(int x, int y, int *z)
{
   *z = x-y;
   return x+y;
}

void add4(int x, int *y)
{
   *y += x;
}
