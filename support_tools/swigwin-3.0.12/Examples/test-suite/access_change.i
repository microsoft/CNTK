%module access_change

// test access changing from protected to public

%inline %{

template<typename T> class Base {
public:
  virtual ~Base() {}
  virtual int *PublicProtectedPublic1() { return 0; }
  int *PublicProtectedPublic2() { return 0; }
  virtual int *PublicProtectedPublic3() { return 0; }
  int *PublicProtectedPublic4() { return 0; }
protected:
  virtual int * WasProtected1() { return 0; }
  int * WasProtected2() { return 0; }
  virtual int * WasProtected3() { return 0; }
  int * WasProtected4() { return 0; }
};

template<typename T> class Derived : public Base<T> {
public:
  int * WasProtected1() { return 0; }
  int * WasProtected2() { return 0; }
  using Base<T>::WasProtected3;
  using Base<T>::WasProtected4;
protected:
  virtual int *PublicProtectedPublic1() { return 0; }
  int *PublicProtectedPublic2() { return 0; }
  using Base<T>::PublicProtectedPublic3;
  using Base<T>::PublicProtectedPublic4;
};

template<typename T> class Bottom : public Derived<T> {
public:
  int * WasProtected1() { return 0; }
  int * WasProtected2() { return 0; }
  using Base<T>::WasProtected3;
  using Base<T>::WasProtected4;
  int *PublicProtectedPublic1() { return 0; }
  int *PublicProtectedPublic2() { return 0; }
  int *PublicProtectedPublic3() { return 0; }
  int *PublicProtectedPublic4() { return 0; }
};
%}

%template(BaseInt) Base<int>;
%template(DerivedInt) Derived<int>;
%template(BottomInt) Bottom<int>;


