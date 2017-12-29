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
  
  if (TYPE($input) != T_ARRAY) {
    SWIG_exception(SWIG_ValueError, "Expected an array");
  }
  $1 = RARRAY_LEN($input);
  if ($1 == 0) {
    SWIG_exception(SWIG_ValueError, "List must contain at least 1 element");
  }
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++) {
    VALUE   s = rb_ary_entry($input,i);
    if (TYPE(s) != T_STRING) {
      free($2);
      SWIG_exception(SWIG_ValueError, "List items must be strings");
    }
    $2[i] = StringValuePtr(s);
  }
  $2[i] = 0;
}

%typemap(freearg) (int argc, char *argv[]) {
  free($2);
}

extern int gcdmain(int argc, char *argv[]);

%typemap(in) (char *bytes, int len) {
  if (TYPE($input) != T_STRING) {
    SWIG_exception(SWIG_ValueError, "Expected a string");
  }
  $1 = StringValuePtr($input);
  $2 = RSTRING_LEN($input);
}

extern int count(char *bytes, int len, char c);


/* This example shows how to wrap a function that mutates a string */

%typemap(in) (char *str, int len) {
  char *temp;
  if (TYPE($input) != T_STRING) {
    SWIG_exception(SWIG_ValueError,"Expected a string");
  }
  temp = StringValuePtr($input);
  $2 = RSTRING_LEN($input);
  $1 = (char *) malloc($2+1);
  memmove($1,temp,$2);
}

/* Return the mutated string as a new object.  */

%typemap(argout, fragment="output_helper") (char *str, int len) {
   VALUE o;
   o = rb_str_new($1,$2);
   $result = output_helper($result,o);
   free($1);
}   

extern void capitalize(char *str, int len);

/* A multi-valued constraint.  Force two arguments to lie
   inside the unit circle */

%typemap(check) (double cx, double cy) {
   double a = $1*$1 + $2*$2;
   if (a > 1.0) {
	SWIG_exception(SWIG_ValueError,"$1_name and $2_name must be in unit circle");
   }
}

extern void circle(double cx, double cy);


