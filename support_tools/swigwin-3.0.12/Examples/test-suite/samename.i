%module samename

#if !(defined(SWIGCSHARP) || defined(SWIGJAVA) || defined(SWIGD))
class samename {
 public:
  void do_something() {
    // ...
  }
};
#endif

%{

class samename {
 public:
  void do_something() {
    // ...
  }
};

%}

