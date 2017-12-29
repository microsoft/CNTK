%module null_pointer

%warnfilter(SWIGWARN_PARSE_KEYWORD) func; // 'func' is a Go keyword, renamed as 'Xfunc'

%inline {
  struct A {};
  
  bool func(A* a) {
    return !a;
  }

  A* getnull() {
    return 0;
  }
}

