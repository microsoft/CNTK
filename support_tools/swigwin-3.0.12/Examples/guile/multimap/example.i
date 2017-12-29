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

%typemap(in) (int argc, char *argv[]) %{
  scm_t_array_handle handle;
  size_t i;
  size_t lenp;
  ssize_t inc;
  const SCM *v;
  if (!(SCM_NIMP($input) && scm_is_vector($input))) {
    SWIG_exception(SWIG_ValueError, "Expecting a vector");
    return 0;
  }
  v = scm_vector_elements($input, &handle, &lenp, &inc);
  $1 = (int)lenp;
  if ($1 == 0) {
    SWIG_exception(SWIG_ValueError, "Vector must contain at least 1 element");
  }
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++, v +=inc ) {
    if (!(SCM_NIMP(*v) && scm_is_string(*v))) {
      free($2);
      SWIG_exception(SWIG_ValueError, "Vector items must be strings");
      return 0;
    }
    $2[i] = scm_to_locale_string(*v);
  }
  $2[i] = 0;
  scm_array_handle_release (&handle);
%}

%typemap(freearg) (int argc, char *argv[]) %{
  for (i = 0; i < $1; i++) {
    free($2[i]);
  }
  free($2);
%}

extern int gcdmain(int argc, char *argv[]);

%typemap(in) (char *bytes, int len) %{
  if (!(SCM_NIMP($input) && scm_is_string($input))) {
    SWIG_exception(SWIG_ValueError, "Expecting a string");
  }
  $1 = scm_to_locale_string($input);
  $2 = scm_c_string_length($input);
%}

%typemap(freearg) (char *bytes, int len) %{
  free($1);
%}

extern int count(char *bytes, int len, char c);

/* This example shows how to wrap a function that mutates a string */

%typemap(in) (char *str, int len) {
  size_t temp;
  $1 = SWIG_Guile_scm2newstr($input,&temp);
  $2 = temp;
}

/* Return the mutated string as a new object.  */

%typemap(argout) (char *str, int len) {
  SWIG_APPEND_VALUE(scm_from_locale_stringn($1,$2));
  if ($1) SWIG_free($1);
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


