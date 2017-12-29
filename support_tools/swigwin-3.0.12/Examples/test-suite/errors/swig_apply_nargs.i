%module xxx

%typemap(in) (char *str, int len) {
}

%apply (char *str, int len) { int x };
