%module profiletest

%inline %{

  class A  {
  public:
    A() {}
  };

  class B
  { 
    A aa;
  public: 
    B() {} 
    A fn(const A* a) {
      return *a;
    }

    int fi(int a) {
      return a;
    }

    int fj(const A* a) {
      return 10;
    }

    B* fk(int i) {
      return this;
    }

    const char* fl(int i) {
      return "hello";
    }

    const char* fs(const char *s) {
      return s;
    }

    int fi(int a, int) {
      return a;
    }

    int fi(char *) {
      return 1;
    }

    int fi(double) {
      return 2;
    }

    int fi(A *a) {
      return 3;
    }

    int fi(int a, int, int) {
      return a;
    }

    int fi(int a, int, int, int) {
      return a;
    }

  };

%}
