%module abstract_access

%warnfilter(SWIGWARN_LANG_DIRECTOR_ABSTRACT) A;

%inline %{
class A {
public:
  virtual ~A()
  {
  }

private:
	virtual int x() = 0;
protected:
	virtual int y() = 0;
public:
	virtual int z() = 0;
	int do_x() { return x(); }
};

class B : public A {
private:
	virtual int x() { return y(); }
};

class C : public B {
protected:
	virtual int y() { return z(); }
};

class D : public C {
private:
	virtual int z() { return 1; }
};

%}
