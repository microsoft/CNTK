/* File : example.i */
%module example

%{
extern int gcd(int x, int y);
extern int gcdmain(int argc, char *argv[]);
extern int count(char *bytes, int len, char c);
extern void capitalize (char *str, int len);
extern void circle (double cx, double cy);
extern int squareCubed (int n, int *OUTPUT);
%}

extern int    gcd(int x, int y);

%typemap(gotype) (int argc, char *argv[]) "[]string"

%typemap(in) (int argc, char *argv[])
%{
  {
    int i;
    _gostring_* a;

    $1 = $input.len;
    a = (_gostring_*) $input.array;
    $2 = (char **) malloc (($1 + 1) * sizeof (char *));
    for (i = 0; i < $1; i++) {
      _gostring_ *ps = &a[i];
      $2[i] = (char *) ps->p;
    }
    $2[i] = NULL;
  }
%}

%typemap(argout) (int argc, char *argv[]) "" /* override char *[] default */

%typemap(freearg) (int argc, char *argv[])
%{
  free($2);
%}

extern int gcdmain(int argc, char *argv[]);

%typemap(gotype) (char *bytes, int len) "string"

%typemap(in) (char *bytes, int len)
%{
  $1 = $input.p;
  $2 = $input.n;
%}

extern int count(char *bytes, int len, char c);

/* This example shows how to wrap a function that mutates a c string. A one
 * element Go string slice is used so that the string can be returned
 * modified.
 */

%typemap(gotype) (char *str, int len) "[]string"

%typemap(in) (char *str, int len)
%{
  {
    _gostring_ *a;
    char *p;
    int n;

    a = (_gostring_*) $input.array;
    p = a[0].p;
    n = a[0].n;
    $1 = malloc(n + 1);
    $2 = n;
    memcpy($1, p, n);
  }
%}

/* Return the mutated string as a modified element in the array. */
%typemap(argout,fragment="AllocateString") (char *str, int len)
%{
  {
    _gostring_ *a;

    a = (_gostring_*) $input.array;
    a[0] = Swig_AllocateString($1, $2);
  }
%}

%typemap(goargout,fragment="CopyString") (char *str, int len)
%{
	$input[0] = swigCopyString($input[0])
%}

%typemap(freearg) (char *str, int len)
%{
  free($1);
%}

extern void capitalize(char *str, int len);

/* A multi-valued constraint.  Force two arguments to lie
   inside the unit circle */

%typemap(check) (double cx, double cy)
%{
  {
     double a = $1*$1 + $2*$2;
     if (a > 1.0) {
       _swig_gopanic("$1_name and $2_name must be in unit circle");
       return;
     }
  }
%}

extern void circle(double cx, double cy);


