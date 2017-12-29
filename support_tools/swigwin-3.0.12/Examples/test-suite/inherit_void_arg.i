%module inherit_void_arg

%inline %{

class A {
public:
        virtual ~A() { }
  
	virtual void f(void) = 0;
};

class B : public A {
public:
        void f() { }
};

%}

