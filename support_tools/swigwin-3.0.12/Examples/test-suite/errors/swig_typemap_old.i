%module xxx

%typemap(in) int x {
   $source;
   $target;
}
