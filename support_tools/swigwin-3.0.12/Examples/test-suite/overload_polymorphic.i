%module overload_polymorphic

%inline %{

class Base {
public:
	Base(){}
	virtual ~Base(){}
};

class Derived : public Base {
public:
	Derived(){}
	virtual ~Derived(){}
};



int test(Base* base){ return 0;}
int test(int hello){ return 1; }

class Unknown;
int test2(Unknown* unknown) { return 0; }
int test2(Base* base) { return 1; }

%}
