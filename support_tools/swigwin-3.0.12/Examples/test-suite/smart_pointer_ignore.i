%module smart_pointer_ignore


%ignore Derived;

%inline %{
class Base {
  public:
    void baseMethod() {}
};

class Derived : public Base {
  public:
    void derivedMethod() {}
};

template <class T>
class Ptr {
public:
  Ptr(T *t) : ptr(t) {}
  T * operator->() const { return ptr; }
private:
  T *ptr;
};
%}

%template(DerivedPtr) Ptr<Derived>;

%inline %{
Ptr<Derived> makeDerived() {
  return Ptr<Derived>(new Derived());
}
%}
