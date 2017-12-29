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
  Scheme_Object **elms;
  if (!SCHEME_VECTORP($input)) {
    scheme_wrong_type("$name","vector",$argnum,argc,argv);
  }
  $1 = SCHEME_VEC_SIZE($input);
  elms = SCHEME_VEC_ELS($input);
  if ($1 == 0) {
    scheme_wrong_type("$name","vector",$argnum,argc,argv);
  }
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++) {
    if (!SCHEME_STRINGP(elms[i])) {
      free($2);
      scheme_wrong_type("$name","vector",$argnum,argc,argv);      
    }
    $2[i] = SCHEME_STR_VAL(elms[i]);
  }
  $2[i] = 0;
}

%typemap(freearg) (int argc, char *argv[]) {
  free($2);
}
extern int gcdmain(int argc, char *argv[]);

%typemap(in) (char *bytes, int len) {
  if (!SCHEME_STRINGP($input)) {
     scheme_wrong_type("$name","string",1,argc,argv);
  }	
  $1 = SCHEME_STR_VAL($input);
  $2 = SCHEME_STRLEN_VAL($input);
}

extern int count(char *bytes, int len, char c);


/* This example shows how to wrap a function that mutates a string */

%typemap(in) (char *str, int len) {
  if (!SCHEME_STRINGP($input)) {
     scheme_wrong_type("$name","string",1,argc,argv);
  }	
  $2 = SCHEME_STRLEN_VAL($input);
  $1 = (char *) malloc($2+1);
  memmove($1,SCHEME_STR_VAL($input),$2);
}

/* Return the mutated string as a new object.  */

%typemap(argout) (char *str, int len) {
   Scheme_Object *s;
   s = scheme_make_sized_string($1,$2,1);
   SWIG_APPEND_VALUE(s);
   free($1);
}   

extern void capitalize(char *str, int len);

/* A multi-valued constraint.  Force two arguments to lie
   inside the unit circle */

%typemap(check) (double cx, double cy) {
   double a = $1*$1 + $2*$2;
   if (a > 1.0) {
	SWIG_exception(SWIG_ValueError,"$1_name and $2_name must be in unit circle");
        return NULL;
   }
}

extern void circle(double cx, double cy);


