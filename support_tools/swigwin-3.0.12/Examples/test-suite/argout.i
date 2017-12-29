/* This interface file checks how well SWIG handles passing data back
   through arguments WITHOUT returning it separately; for the cases where
   maybe multiple values are passed by reference and all want changing */

%module argout

%include cpointer.i
%pointer_functions(int,intp);

%inline %{
// returns old value
int incp(int *value) {
  return (*value)++;
}

// returns old value
int incr(int &value) {
  return value++;
}

typedef int & IntRef;
// returns old value
int inctr(IntRef value) {
  return value++;
}

// example of the old DB login type routines where you keep
// a void* which it points to its opaque struct when you login
// So login function takes a void**
void voidhandle(void** handle) {
  *handle=(void*)"Here it is";
}
char * handle(void* handle) {
  return (char *)handle;
}

%}
