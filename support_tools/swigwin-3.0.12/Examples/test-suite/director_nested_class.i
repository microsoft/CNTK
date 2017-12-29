%module(directors="1") director_nested_class


%feature("director") DirectorOuter::DirectorInner;
%feature("director") DirectorOuter::DirectorInner::DirectorInnerInner;

%inline %{
struct DirectorOuter {
  struct DirectorInner {
    virtual ~DirectorInner() {}
    virtual int vmethod(int input) const = 0;
    struct DirectorInnerInner {
      DirectorInnerInner(DirectorInner *din = 0) {}
      virtual ~DirectorInnerInner() {}
      virtual int innervmethod(int input) const = 0;
    };
  };
  static int callMethod(const DirectorInner &di, int value) {
    return di.vmethod(value);
  } 
  static int callInnerInnerMethod(const DirectorInner::DirectorInnerInner &di, int value) {
    return di.innervmethod(value);
  } 
};
%}
