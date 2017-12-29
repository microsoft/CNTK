%module foolib
%{
static int foo( char **buf ) {
  *buf = "string from c";
  return 0;
}
%}

%typemap(in,numinputs=0) char **buf (char *temp) {
    $1 = &temp;
}
%typemap(argout) char **buf {
    swig_result = caml_list_append(swig_result,caml_val_string((char *)*$1));
}

int foo( char **buf );
