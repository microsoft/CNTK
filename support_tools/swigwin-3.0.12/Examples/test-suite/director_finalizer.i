%module(directors="1") director_finalizer

%warnfilter(SWIGWARN_GO_NAME_CONFLICT);                       /* Ignoring 'deleteFoo' due to Go name ('DeleteFoo') conflict with '~Foo' */

%{

int status = 0;

class Foo {
public:
	virtual ~Foo() { orStatus(1); }
	virtual void orStatus(int x) { status |= x; }
};

void deleteFoo(Foo *f) {
	delete f;
}

Foo *launder(Foo *f) {
	return f;
}

int getStatus() {
	return status;
}

void resetStatus() {
	status = 0;
}

%}

%feature("director") Foo;

class Foo {
public:
	virtual ~Foo();
	virtual void orStatus(int x);
};

void deleteFoo(Foo *f);
int getStatus();
Foo *launder(Foo *f);
void resetStatus();

