%module lua_inherit_getitem

%inline %{

class CBase {
public:
  const char* Foo(void) {
    return "CBase::Foo";
  }
};

class CDerived : public CBase {
public:
  void *__getitem(const char *name) const {
    return NULL;
  }
};

%}
