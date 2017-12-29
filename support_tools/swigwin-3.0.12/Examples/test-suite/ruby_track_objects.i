%module ruby_track_objects

%include typemaps.i

%trackobjects Foo;

%newobject Bar::get_new_foo;

%typemap(in, numinputs=0) Foo** foo (Foo *temp) {
	/* %typemap(in, numinputs=0) Foo** foo */
	$1 = &temp;
}

%typemap(argout) Foo** foo {
	/* %typemap(argout) Foo** foo */
	$result = SWIG_NewPointerObj((void *) *$1, $*1_descriptor, 0);
}	

%apply SWIGTYPE *DISOWN {Foo* ownedFoo};


%trackobjects ItemA;
%trackobjects ItemB;

%inline %{

class Foo
{
public:
	Foo() {}
	~Foo() {}

	/* Helper method that can be called from Ruby that checks
	   that two Ruby objects are pointing to the same underlying
		C++ object */
	bool cpp_equal(const Foo* other)
	{
		return (this == other);
	}

	/* Just a simple method to call on Foo*/
	const char* say_hello()
	{
		return "Hello";
	}
};


class Bar
{
private:
	Foo* owned_;
	Foo* unowned_;
public:
	Bar(): owned_(new Foo), unowned_(0)
	{
	}

	~Bar()
	{
		delete owned_;
	}

	/* Test that track objects works with %newobject */
	static Foo* get_new_foo()
	{
		return new Foo;
	}

	/* Test the same foo Ruby object is created each time */
	Foo* get_owned_foo()
	{
		return owned_;
	}

	/* Test that track objects works with argout parameters.*/
	void get_owned_foo_by_argument(Foo** foo)
	{
		*foo = owned_;
	}

	/* Test that track objects works with the DISOWN typemap.*/
	void set_owned_foo(Foo* ownedFoo)
	{
		delete owned_;
		owned_ = ownedFoo;
	}

	Foo* get_unowned_foo()
	{
		return unowned_;
	}

	void set_unowned_foo(Foo* foo)
	{
		unowned_ = foo;
	}
};

class ItemA
{
};

class ItemB: public ItemA
{
public:
};

ItemB* downcast(ItemA* item)
{
	return static_cast<ItemB*>(item);
}

class Factory
{
public:
	Factory() {}

	ItemA* createItem()
	{
		return new ItemB;
	}
};

%}
