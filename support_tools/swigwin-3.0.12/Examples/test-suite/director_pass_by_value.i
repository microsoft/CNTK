%module(directors="1") director_pass_by_value
%director DirectorPassByValueAbstractBase;

%inline %{
class PassedByValue {
  int val;
public:
  PassedByValue() { val = 0x12345678; }
  int getVal() { return val; }
};

int doSomething(int x) {
  int yy[256];
  yy[0] =0x9876;
  return yy[0];
}

class DirectorPassByValueAbstractBase {
public:
  virtual void virtualMethod(PassedByValue pbv) = 0;
  virtual ~DirectorPassByValueAbstractBase () {}
};

class Caller {
public:
  void call_virtualMethod(DirectorPassByValueAbstractBase &f) {
    f.virtualMethod(PassedByValue());
  }
};
%}
