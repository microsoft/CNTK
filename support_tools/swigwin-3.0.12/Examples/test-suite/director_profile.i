%module(directors="1") director_profile

%include std_string.i

%feature("director") B;


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

    virtual A fn(const A* a) {
      return *a;
    }

    virtual int vfi(int a) {
      return a + 1;
    }

    int fi(int a) {
      return vfi(a);
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

    virtual ~B()
    {
    }

    static B* get_self(B *b) 
    {
      return b;
    }

    virtual std::string vfs(const std::string& a) {
      return a;
    }
    
    std::string fs(const std::string& a) {
      return vfs(a);
    }
    
  };

%}
