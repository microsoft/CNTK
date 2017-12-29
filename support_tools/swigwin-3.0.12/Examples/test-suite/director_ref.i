%module(directors="1") director_ref

%{
#include <string>

class Foo {
public:
        Foo(int i = -1) : count(0) {}
	virtual void OnDelete() {}
	virtual ~Foo() {}
	virtual std::string Msg(std::string msg = "default") { return "Foo-" + msg; }

	std::string GetMsg() { return Msg(); }
	std::string GetMsg(std::string msg) { return Msg(msg); }

	void Ref() { ++count; }
	void Unref() { --count; if (count == 0) { OnDelete(); delete this; } }
	int GetRefCount() { return count; }
private:
	int count;
};

class FooPtr {
public:
        FooPtr(Foo* f = NULL) : my_f(f) { if (my_f) { my_f->Ref(); } }
        ~FooPtr() { if (my_f) { my_f->Unref(); } }
        void Reset(Foo* f = NULL) {
		if (f) { f->Ref(); }
		if (my_f) { my_f->Unref(); }
		my_f = f;
	}
	int GetOwnedRefCount() {
		if (my_f) { return my_f->GetRefCount(); }
		return 0;
	}

private:
        Foo* my_f;
};

%}

%include <std_string.i>

%feature("director") Foo;
%feature("ref") Foo "$this->Ref();"
%feature("unref") Foo "$this->Unref();"

class Foo {
public:
        Foo(int i = -1) : count(0) {}
	virtual void OnDelete() {}
	virtual ~Foo() {}
	virtual std::string Msg(std::string msg = "default") { return "Foo-" + msg; }

	std::string GetMsg() { return Msg(); }
	std::string GetMsg(std::string msg) { return Msg(msg); }

	void Ref() { ++count; }
	void Unref() { --count; if (count == 0) { OnDelete(); delete this; } }
	int GetRefCount() { return count; }
private:
	int count;
};

class FooPtr {
public:
        FooPtr(Foo* f = NULL) : my_f(f) { if (my_f) { my_f->Ref(); } }
        ~FooPtr() { if (my_f) { my_f->Unref(); } }
        void Reset(Foo* f = NULL) {
		if (f) { f->Ref(); }
		if (my_f) { my_f->Unref(); }
		my_f = f;
	}
	int GetOwnedRefCount() {
		if (my_f) { return my_f->GetRefCount(); }
		return 0;
	}

private:
        Foo* my_f;
};
