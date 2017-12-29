%module virtual_vs_nonvirtual_base;
// Regression test for SF#3124665.
%inline {

class SimpleVirtual
{
	public:
		virtual int implementMe() = 0;
		virtual ~SimpleVirtual() {}
};

class SimpleNonVirtual
{
	public:
		int dummy() { return 0; }
		virtual ~SimpleNonVirtual() {}
};

class SimpleReturnClass
{
	public:
		SimpleReturnClass(int i) : value(i) {};
		int get() const { return value; }
	private:
		int value;
};

class SimpleClassFail : public SimpleVirtual
{
	public:
		SimpleClassFail() : inner(10) {}
		SimpleReturnClass getInner() { return inner; }
		
		virtual int implementMe() { return 0; }
	private:
		SimpleReturnClass inner;
};

class SimpleClassWork : public SimpleNonVirtual
{
	public:
		SimpleClassWork() : inner(10) {}
		SimpleReturnClass getInner() { return inner; }
		
		virtual int implementMe() { return 0; }
	private:
		SimpleReturnClass inner;
};

}
