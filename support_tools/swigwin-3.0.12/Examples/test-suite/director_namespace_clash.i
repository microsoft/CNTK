%module(directors="1") director_namespace_clash

%rename(GreatOne) One::Great;

%feature("director");

%inline %{
namespace One {
  struct Great {
    virtual void superb(int a) {}
    virtual ~Great() {}
  };
}
namespace Two {
  struct Great {
    virtual void excellent() {}
    virtual ~Great() {}
  };
}
%}

