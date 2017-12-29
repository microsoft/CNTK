%module octave_cell_deref

%inline {
  bool func(const char* s) {
    return !strcmp("hello",s);
  }

 Cell func2() {
   Cell c(1,2);
   c(0) = "hello";
   c(1) = 4;
   return c;
 } 
}

