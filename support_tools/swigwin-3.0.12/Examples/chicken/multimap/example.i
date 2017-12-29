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

%include exception.i
%include typemaps.i

extern int    gcd(int x, int y);

%typemap(in) (int argc, char *argv[]) {
  int i;
  if (!C_swig_is_vector ($input)) {
    swig_barf (SWIG_BARF1_BAD_ARGUMENT_TYPE, "Argument $input is not a vector");
  }
  $1 = C_header_size ($input);
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++) {
    C_word o = C_block_item ($input, i);
    if (!C_swig_is_string (o)) {
      char err[50];
      free($2);
      sprintf (err, "$input[%d] is not a string", i);
      swig_barf (SWIG_BARF1_BAD_ARGUMENT_TYPE, err);
    }
    $2[i] = C_c_string (o);
  }
  $2[i] = 0;
}

%typemap(freearg) (int argc, char *argv[]) {
  free($2);
}
extern int gcdmain(int argc, char *argv[]);

%typemap(in) (char *bytes, int len) {
  if (!C_swig_is_string ($input)) {
    swig_barf (SWIG_BARF1_BAD_ARGUMENT_TYPE, "Argument $input is not a string");
  }	
  $1 = C_c_string ($input);
  $2 = C_header_size ($input);
}

extern int count(char *bytes, int len, char c);


/* This example shows how to wrap a function that mutates a string */

%typemap(in) (char *str, int len) 
%{  if (!C_swig_is_string ($input)) {
    swig_barf (SWIG_BARF1_BAD_ARGUMENT_TYPE, "Argument $input is not a string");
  }
  $2 = C_header_size ($input);
  $1 = (char *) malloc ($2+1);
  memmove ($1, C_c_string ($input), $2);
%}

/* Return the mutated string as a new object.  Notice the if MANY construct ... they must be at column 0. */

%typemap(argout) (char *str, int len) (C_word *scmstr) 
%{  scmstr = C_alloc (C_SIZEOF_STRING ($2));
  SWIG_APPEND_VALUE(C_string (&scmstr, $2, $1));
  free ($1);
%}

extern void capitalize (char *str, int len);

/* A multi-valued constraint.  Force two arguments to lie
   inside the unit circle */

%typemap(check) (double cx, double cy) {
  double a = $1*$1 + $2*$2;
  if (a > 1.0) {
    SWIG_exception (SWIG_ValueError, "cx and cy must be in unit circle");
  }
}

extern void circle (double cx, double cy);

/* Test out multiple return values */

extern int squareCubed (int n, int *OUTPUT);
%{
/* Returns n^3 and set n2 to n^2 */
int squareCubed (int n, int *n2) {
  *n2 = n * n;
  return (*n2) * n;
};
%}
