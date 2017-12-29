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

extern int    gcd(int x, int y);

%typemap(arginit) (int argc, char *argv[]) "$2 = 0;";

%typemap(in) (int argc, char *argv[]) {
  Tcl_Obj **listobjv = 0;
  int i;
  if (Tcl_ListObjGetElements(interp,$input, &$1, &listobjv) == TCL_ERROR) {
    SWIG_exception(SWIG_ValueError,"Expected a list");
    return TCL_ERROR;
  }
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++) {
    $2[i] = Tcl_GetStringFromObj(listobjv[i],0);
  }
  $2[i] = 0;
}

%typemap(freearg) char *argv[] {
  if ($1) {
    free($1);
  }
}

extern int gcdmain(int argc, char *argv[]);

%typemap(in) (char *bytes, int len) {
  $1 = Tcl_GetStringFromObj($input,&$2);
}

extern int count(char *bytes, int len, char c);


/* This example shows how to wrap a function that mutates a string */

%typemap(in) (char *str, int len) {
  char *temp;
  temp = Tcl_GetStringFromObj($input,&$2);
  $1 = (char *) malloc($2+1);
  memmove($1,temp,$2);
}

/* Return the mutated string as a new object.   */

%typemap(argout) (char *str, int len) {
 Tcl_Obj *o;
 o = Tcl_NewStringObj($1,$2);
 Tcl_ListObjAppendElement(interp,$result,o);
 free($1);
}   

extern void capitalize(char *str, int len);


/* A multi-valued constraint.  Force two arguments to lie
   inside the unit circle */

%typemap(check) (double cx, double cy) {
   double a = $1*$1 + $2*$2;
   if (a > 1.0) {
	SWIG_exception(SWIG_ValueError,"$1_name and $2_name must be in unit circle");
        return TCL_ERROR;
   }
}

extern void circle(double cx, double cy);
